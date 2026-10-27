"""Continue one existing experiment with two GPU replicas and a fixed global batch."""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from research.qwen35_omega.cache import atomic_save
from .cache import auxiliary, identity, make_batch, native
from .config import ARMS, CACHE, RUN, TRAINING_SEEDS
from .runtime import load_model, objective, read_rows, restore_trainable, stage, trainable_state
from .train import epoch_order, gradients, microbatch, optimizer


def local_group(group, rank, world):
    """Balanced contiguous pieces preserve the original global example group."""
    begin = len(group) * rank // world
    end = len(group) * (rank + 1) // world
    return group[begin:end]


def gradient_scale(part_size, global_size, world):
    # DDP averages gradients across ranks; compensate to recover a per-example
    # global mean, including the final uneven 11-question group.
    return world * part_size / global_size


class FeaturePrefetch:
    """Warm existing bounded CPU feature caches while GPUs process earlier groups."""
    def __init__(self, arm, workers):
        self.kind = ARMS[arm]
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='feature')

    def warm(self, row):
        for digest in row['image_hashes']:
            native(digest)
        if self.kind:
            auxiliary(self.kind, row['feature_key'])

    def submit(self, rows):
        return [self.pool.submit(self.warm, row) for row in rows]

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def rng_state():
    return {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}


def restore_rng(checkpoint, rank):
    states = checkpoint.get('rank_rng_states')
    if states and rank < len(states):
        value = states[rank]
    elif rank == 0:
        value = {key: checkpoint[key] for key in ('torch_rng', 'python_rng', 'numpy_rng')}
        value['cuda_rng'] = checkpoint['cuda_rng'][0]
    else:
        # One experiment seed remains 42; a newly added rank needs its own
        # dropout RNG stream. A world-size change is not bitwise continuation.
        stream = 42 + 3000 + rank * 100003 + checkpoint['global_step']
        random.seed(stream); np.random.seed(stream); torch.manual_seed(stream)
        torch.cuda.manual_seed(stream)
        return
    torch.set_rng_state(value['torch_rng'])
    torch.cuda.set_rng_state(value['cuda_rng'])
    random.setstate(value['python_rng']); np.random.set_state(value['numpy_rng'])


def read_checkpoint(folder):
    name = json.loads((folder/'latest.json').read_text())['file']
    return torch.load(folder/name, map_location='cpu', weights_only=False, mmap=True)


def validate_configuration(folder):
    configuration = json.loads((folder/'configuration.json').read_text())
    if configuration['arm'] != 'O2' or configuration['seed'] not in TRAINING_SEEDS:
        raise RuntimeError('Only the current O2 single-seed experiment may continue')
    if configuration['identity'] != identity():
        raise RuntimeError('Frozen feature identity changed')
    split = RUN/'splits'/f"{configuration['subset']}.jsonl"
    if hashlib.sha256(split.read_bytes()).hexdigest() != configuration['split_sha256']:
        raise RuntimeError('Training split changed')
    # Continue with the exact original model, objective, batching, and optimizer
    # helper sources; the distributed orchestration is separately archived.
    for name, expected in configuration['code_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Original training semantics changed: {name}')
    warm = Path(configuration['warm_checkpoint'])
    if warm.stat().st_size != configuration['warm_size'] or warm.stat().st_mtime_ns != configuration['warm_mtime_ns']:
        raise RuntimeError('Common warm checkpoint changed')
    return configuration


def backward_group(ddp, wrapped, group, arm, answer_ids, rank, world):
    rows = local_group(group, rank, world)
    if not rows:
        raise RuntimeError('Each rank must receive at least one example')
    bucket = math.ceil(max(row['input_ids'].numel() for row in group)/128)*128
    size = min(microbatch(bucket), len(rows))
    stats = defaultdict(float)
    for begin in range(0, len(rows), size):
        part = rows[begin:begin+size]
        sync = begin + size >= len(rows)
        with nullcontext() if sync else ddp.no_sync():
            batch, context = make_batch(part, arm)
            loss, values = objective(ddp, wrapped, batch, context, answer_ids)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite distributed loss')
            (loss*gradient_scale(len(part), len(group), world)).backward()
            for key, value in dict(loss=loss.item(), **values).items():
                stats[key] += value*len(part)/len(group)
            del loss, batch, context
    keys = sorted(stats)
    values = torch.tensor([stats[key] for key in keys], device='cuda', dtype=torch.float64)
    dist.all_reduce(values)
    return dict(zip(keys, values.cpu().tolist())), size, bucket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--handoff-dir', type=Path, required=True)
    parser.add_argument('--prefetch-workers', type=int, default=16)
    parser.add_argument('--cpu-threads', type=int, default=32)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    args = parser.parse_args()
    if args.checkpoint_interval<1:raise ValueError('Checkpoint interval must be positive')
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 2:
        raise RuntimeError('This continuation is validated for exactly two GPUs')
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(args.cpu_threads); torch.set_num_interop_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=torch.device('cuda', local_rank))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=30))
    args.handoff_dir.mkdir(parents=True, exist_ok=True)
    configuration = validate_configuration(args.output)
    if (args.output/'complete.json').exists():
        if rank == 0: print('The declared experiment is already complete', flush=True)
        dist.destroy_process_group(); return
    checkpoint = read_checkpoint(args.output)
    if checkpoint['stage'] != 'joint':
        raise RuntimeError('Expected the existing run to be in its joint stage')
    print(f'rank={rank} local_cuda={local_rank} loading checkpoint step={checkpoint["global_step"]}', flush=True)
    tokens = torch.load(CACHE/'tokens.pt', map_location='cpu', weights_only=True, mmap=True)
    rows = read_rows(RUN/'splits'/f"{configuration['subset']}.jsonl")
    records = [tokens[row['id']] for row in rows]; del tokens, rows
    answer_ids = list(json.loads((CACHE/'tokens_full_train_0_complete.json').read_text())['answer_token_ids'].values())
    model, wrapped = load_model(configuration['arm'], configuration['seed'])
    if {p.device.index for p in model.parameters()} != {local_rank}:
        raise RuntimeError('Model parameters are not confined to the rank GPU')
    print(f'rank={rank} model loaded on cuda:{local_rank}; starting distributed backward probes', flush=True)
    stage(model, wrapped, 'joint'); restore_trainable(model, checkpoint['adapter'])
    opt = optimizer(model, 'joint', configuration['adapter_lr']); opt.load_state_dict(checkpoint['optimizer'])
    ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                  gradient_as_bucket_view=True, bucket_cap_mb=64)
    prefetch = FeaturePrefetch(configuration['arm'], args.prefetch_workers)
    # Run real forward/backward probes before asking the existing job to stop.
    # Their gradients/RNG effects are discarded when the fresh handoff reloads.
    probes = [max(records, key=lambda row: row['input_ids'].numel())]
    probes.append(max((row for row in records if row['input_ids'].numel() <= 1536),
                      key=lambda row: row['input_ids'].numel()))
    torch.cuda.reset_peak_memory_stats()
    for record in probes:
        group = [record]*32
        for future in prefetch.submit(local_group(group, rank, world)): future.result()
        opt.zero_grad(set_to_none=True)
        backward_group(ddp, wrapped, group, configuration['arm'], answer_ids, rank, world)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    opt.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated()/2**30
    if peak >= 240: raise RuntimeError(f'Preflight memory guard exceeded: {peak:.2f} GiB')
    details = [None]*world
    dist.all_gather_object(details, {'rank':rank, 'gpu':torch.cuda.get_device_name(),
                                   'peak_allocated_gib':peak, 'preflight_passed':True}, group=control)
    if rank == 0:
        ready = {'world_size':world, 'global_batch':32, 'ranks':details,
                 'probe_checkpoint_step':checkpoint['global_step'], 'cpu_threads_per_rank':args.cpu_threads,
                 'prefetch_workers_per_rank':args.prefetch_workers}
        (args.handoff_dir/'ready.json').write_text(json.dumps(ready, indent=2)+'\n')
        print('TWO_GPU_HANDOFF_READY '+json.dumps(ready), flush=True)
    while not (args.handoff_dir/'go.json').exists(): time.sleep(1)
    dist.barrier(group=control)
    del checkpoint
    checkpoint = read_checkpoint(args.output)
    restore_trainable(model, checkpoint['adapter']); opt.load_state_dict(checkpoint['optimizer'])
    restore_rng(checkpoint, rank)
    global_step = checkpoint['global_step']; stage_step = checkpoint['stage_step']
    start_epoch = checkpoint['epoch'] + int(checkpoint['epoch_complete'])
    start_begin = 0 if checkpoint['epoch_complete'] else checkpoint['next_begin']
    # Capture and preserve the old log before removing its unsaved tail.
    timing = [None]
    if rank == 0:
        log = args.output/'training.jsonl'; backup = args.handoff_dir/'training_before_handoff.jsonl'
        if backup.exists(): raise FileExistsError(backup)
        os.replace(log, backup)
        kept = []; elapsed = 0.
        for line in backup.read_text().splitlines():
            try: event = json.loads(line)
            except ValueError: continue
            if event['global_step'] <= global_step:
                kept.append(line); elapsed = event['elapsed_seconds']
        log.write_text('\n'.join(kept)+'\n'); timing[0] = elapsed
        execution = {'started_utc':datetime.now(timezone.utc).isoformat(), 'world_size':world,
                     'global_batch_size':32, 'local_batch_size':16, 'continued_from_step':global_step,
                     'original_configuration_preserved':True, 'training_seed':configuration['seed'],
                     'rank_rng_note':'Independent dropout streams; world-size migration is not bitwise equivalent to one GPU.',
                     'cpu_threads_per_rank':args.cpu_threads, 'prefetch_workers_per_rank':args.prefetch_workers,
                     'checkpoint_interval':args.checkpoint_interval,
                     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'handoff_directory':str(args.handoff_dir)}
        (args.output/'distributed_execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    dist.broadcast_object_list(timing, src=0, group=control)
    elapsed_before = timing[0]; started = time.monotonic()
    del checkpoint
    def save(epoch, next_begin, epoch_complete=False):
        states = [None]*world
        dist.all_gather_object(states, rng_state(), group=control)
        if rank == 0:
            value = {'adapter':trainable_state(model), 'optimizer':opt.state_dict(), 'epoch':epoch,
                     'stage':'joint', 'next_begin':next_begin, 'stage_step':stage_step,
                     'epoch_complete':epoch_complete, 'global_step':global_step,
                     'torch_rng':states[0]['torch_rng'], 'cuda_rng':[states[0]['cuda_rng']],
                     'python_rng':states[0]['python_rng'], 'numpy_rng':states[0]['numpy_rng'],
                     'rank_rng_states':states, 'world_size':world, 'configuration':configuration}
            name = f'epoch_{epoch}.pt' if epoch_complete else 'progress.pt'
            atomic_save(value, args.output/name)
            tmp = args.output/'latest.tmp.json'; tmp.write_text(json.dumps({'file':name})+'\n')
            os.replace(tmp, args.output/'latest.json')
        dist.barrier(group=control)
    for epoch in range(start_epoch, configuration['joint_epochs']+2):
        order = epoch_order(records, configuration['seed'], epoch)
        begin_at = start_begin if epoch == start_epoch else 0
        pending = deque()
        def enqueue(begin):
            if begin < len(order):
                group = [records[i] for i in order[begin:begin+32]]
                pending.append((begin, group, prefetch.submit(local_group(group, rank, world))))
        for offset in range(3): enqueue(begin_at+32*offset)
        steps = math.ceil(len(order)/32); total = steps*configuration['joint_epochs']
        warmup = max(1, math.ceil(total*.03)); torch.cuda.reset_peak_memory_stats()
        while pending:
            begin, group, futures = pending.popleft()
            for future in futures: future.result()
            enqueue(begin+3*32)
            opt.zero_grad(set_to_none=True)
            factor = (stage_step+1)/warmup if stage_step < warmup else .5*(1+math.cos(math.pi*(stage_step-warmup)/max(1,total-warmup)))
            for g in opt.param_groups: g['lr'] = g['initial_lr']*factor
            values, size, bucket = backward_group(ddp, wrapped, group, configuration['arm'], answer_ids, rank, world)
            per_group = gradients(model)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            opt.step(); stage_step += 1; global_step += 1
            peaks = torch.tensor([torch.cuda.max_memory_allocated()/2**30], device='cuda')
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
            if rank == 0:
                event = dict(epoch=epoch, stage='joint', global_step=global_step, examples=begin+len(group),
                             microbatch=size, bucket=bucket, grad_norm=norm.item(), group_grad_norms=per_group,
                             peak_allocated_gib=peaks.item(), elapsed_seconds=elapsed_before+time.monotonic()-started,
                             world_size=world, global_batch=len(group), **values)
                event.update(wrapped.alignment.last_diagnostics)
                with (args.output/'training.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
                if global_step%10 == 0: print(json.dumps(event), flush=True)
            if global_step%args.checkpoint_interval == 0: save(epoch, begin+len(group))
        save(epoch, len(order), True)
    prefetch.close()
    if rank == 0:
        complete = {'epochs':configuration['joint_epochs']+1, 'global_steps':global_step,
                    'unique_records':len(records), 'joint_epochs':configuration['joint_epochs'], 'tiny_steps':0,
                    'elapsed_seconds':elapsed_before+time.monotonic()-started,
                    'final_checkpoint':f"epoch_{configuration['joint_epochs']+1}.pt", 'world_size':world}
        (args.output/'complete.json').write_text(json.dumps(complete, indent=2)+'\n')
    dist.barrier(group=control); dist.destroy_process_group()


if __name__ == '__main__': main()
