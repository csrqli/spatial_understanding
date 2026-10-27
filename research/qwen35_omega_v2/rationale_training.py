"""One conservative rationale-supervised continuation of completed O2.

This module is prepared as a follow-up candidate. It is not launched while the
matched inference diagnostic is running. Training seed is fixed at 42.
"""
import argparse
from collections import Counter,defaultdict
from contextlib import nullcontext
from datetime import datetime,timedelta,timezone
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
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from research.qwen35_omega.cache import atomic_save
from .cache import make_batch
from .config import CACHE,ROOT,RUN
from .runtime import load_model,read_rows,restore_trainable,stage,trainable_state
from .train import kind
from .train_distributed import FeaturePrefetch,local_group,rng_state

V3=Path(os.environ.get('Q35V3_ROOT',str(ROOT/'results/qwen35_omega_omnispatial/20261026_improve_v3')))


def build_records():
    """Retain all train-side Omni questions; balance static/dynamic SAT QA."""
    rows=read_rows(RUN/'splits/train.jsonl')
    old=torch.load(CACHE/'tokens.pt',map_location='cpu',weights_only=True,mmap=True)
    rationale=torch.load(V3/'data/reasoning_tokens.pt',map_location='cpu',weights_only=True,mmap=True)
    rng=random.Random(42)
    selected=[dict(old[r['id']],original_id=r['id'],variant='qa') for r in rows if r['source']=='omni']
    sat=defaultdict(list)
    for row in rows:
        if row['source']=='sat':sat[row['task_type']].append(row)
    for task,values in sorted(sat.items()):
        values=sorted(values,key=lambda r:r['id']);rng.shuffle(values)
        selected.extend(dict(old[r['id']],original_id=r['id'],variant='qa') for r in values[:1600])
    selected.extend(dict(r,variant='reasoning') for r in rationale.values() if r['training_partition']=='train')
    selected.sort(key=lambda r:r['id'])
    if len({r['original_id'] for r in selected})!=len(selected):raise RuntimeError('Duplicate training questions')
    if any(r['source'] not in ('omni','sat','mind') for r in selected):raise RuntimeError('Unexpected source')
    if any(r['id'].startswith('omni_test/') for r in selected):raise RuntimeError('Test question in training')
    return selected


def repetitions(record):
    if record['variant']=='reasoning':return 2
    if record['source']=='sat':return 1
    return {'Spatial_Interaction':2,'Perspective_Taking':2,'Complex_Logic':4,'Dynamic_Reasoning':12}[record['task_type']]


def order_for_epoch(records,epoch):
    order=[i for i,r in enumerate(records) for _ in range(repetitions(r))]
    random.Random(42+1009*epoch).shuffle(order)
    for begin in range(0,len(order),512):
        order[begin:begin+512]=sorted(order[begin:begin+512],key=lambda i:records[i]['input_ids'].numel())
    return order


def microbatch(length):
    # Conservative initial bounds; GPU preflight must verify these before
    # continuing the full run. Original global batch remains 32.
    return 4 if length<=1024 else 2 if length<=2560 else 1


def implementation_hashes():
    """Reject a resume with a different objective, model, or batching path."""
    files=[Path(__file__).resolve()]
    here=files[0].parent
    files.extend(here/name for name in ['runtime.py','fusion.py','cache.py',
                                      'train.py','train_distributed.py','config.py'])
    files.extend(here.parent/'qwen35_omega'/name for name in
                 ['runtime.py','fusion.py','cache.py','data.py'])
    return {str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def loss_from_logits(logits,target,answer_mask):
    valid=target!=-100
    if not torch.equal((answer_mask&valid).sum(-1),torch.ones(target.shape[0],device=target.device,dtype=torch.long)):
        raise RuntimeError('Exactly one final answer token per question is required')
    ce=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').view_as(target)
    answer=ce[answer_mask].mean()
    rationale=valid&~answer_mask
    # Balance questions, rather than giving long teacher traces more loss mass.
    denominator=rationale.sum(-1)
    if (denominator==0).any():raise RuntimeError('Empty rationale/format supervision')
    reasoning=((ce*rationale).sum(-1)/denominator).mean()
    loss=.5*answer+.5*reasoning
    return loss,{'answer_ce':answer.detach().item(),'rationale_format_ce':reasoning.detach().item(),
                 'answer_accuracy':(logits.argmax(-1)[answer_mask]==target[answer_mask]).float().mean().detach().item(),
                 'supervised_tokens':int(valid.sum())}


def objective(model,wrapped,batch,context,records):
    device=next(model.parameters()).device
    batch={k:v.to(device) for k,v in batch.items()}
    labels=batch.pop('labels')
    shifted=F.pad(labels,(0,1),value=-100)[:,1:]
    indices=torch.where((shifted!=-100).any(0))[0]
    target=shifted[:,indices]
    positions=torch.tensor([labels.shape[1]-r['input_ids'].numel()+r['answer_position']-1 for r in records],device=device)
    answer_mask=indices[None,:]==positions[:,None]
    wrapped.context=context
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
            logits=model(**batch,logits_to_keep=indices,use_cache=False).logits
            return loss_from_logits(logits,target,answer_mask)
    finally:wrapped.context=None


def make_optimizer(model):
    rates={'fusion':2e-5,'merger':1e-6,'lora':1e-6}
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        component=kind(name);decay=p.ndim>1 and not any(s in name for s in ('norm','gate','bias'))
        key=component,decay
        if key not in groups:groups[key]={'params':[],'lr':rates[component],'initial_lr':rates[component],
                     'weight_decay':.01 if decay else 0.,'component':component}
        groups[key]['params'].append(p)
    return torch.optim.AdamW(list(groups.values()),betas=(.9,.95),eps=1e-8)


def self_check():
    """Repeated option-letter tokens inside a rationale must not be answers."""
    torch.manual_seed(42)
    logits=torch.randn(2,6,13,requires_grad=True)
    target=torch.tensor([[3,4,3,7,9,-100],[3,3,5,6,8,9]])
    mask=torch.tensor([[False,False,False,True,False,False],
                       [False,False,False,False,True,False]])
    actual,stats=loss_from_logits(logits,target,mask)
    ce=F.cross_entropy(logits.reshape(-1,13),target.reshape(-1),ignore_index=-100,reduction='none').view(2,6)
    valid=target!=-100;other=valid&~mask
    reference=.5*ce[mask].mean()+.5*((ce*other).sum(-1)/other.sum(-1)).mean()
    torch.testing.assert_close(actual,reference)
    actual.backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    try:loss_from_logits(logits,target,torch.zeros_like(mask))
    except RuntimeError:pass
    else:raise AssertionError('Missing final answer accepted')
    print('Rationale loss, final-answer positions, question weighting and gradients passed.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=V3/'rationale_O2_seed42')
    parser.add_argument('--initialize',type=Path,default=RUN/'full/O2_seed42/epoch_4.pt')
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--checkpoint-interval',type=int,default=25)
    parser.add_argument('--gradient-checkpointing',action='store_true')
    parser.add_argument('--max-steps',type=int,default=0,help='Engineering initial segment; resume the same optimizer state for full continuation')
    parser.add_argument('--check-loss',action='store_true')
    args=parser.parse_args()
    if args.check_loss:self_check();return
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=2:raise RuntimeError('This continuation requires exactly two GPU replicas')
    torch.cuda.set_device(local);torch.set_num_threads(16)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device('cuda',local))
    control=dist.new_group(backend='gloo',timeout=timedelta(minutes=30))
    records=build_records();count=len(order_for_epoch(records,1));steps_per_epoch=math.ceil(count/32)
    configuration={'arm':'O2','seed':42,'epochs':args.epochs,'global_batch':32,'records':len(records),
                   'exposures_per_epoch':count,'planned_steps':steps_per_epoch*args.epochs,
                   'initialize':str(args.initialize),'initial_checkpoint_size':args.initialize.stat().st_size,
                   'sources':dict(Counter(r['source'] for r in records)),
                   'variants':dict(Counter(r['variant'] for r in records)),
                   'implementation_sha256':implementation_hashes(),
                   'reasoning_audit_sha256':hashlib.sha256((V3/'data/reasoning_audit.json').read_bytes()).hexdigest(),
                   'rationale_manifest_sha256':hashlib.sha256((V3/'data/reasoning_manifest.jsonl').read_bytes()).hexdigest(),
                   'selected_questions_sha256':hashlib.sha256(json.dumps([r['id'] for r in records]).encode()).hexdigest(),
                   'rates':{'fusion':2e-5,'merger':1e-6,'lora':1e-6},'gradient_checkpointing':args.gradient_checkpointing,
                   'loss':'0.5 final-answer CE + 0.5 question-mean rationale/format CE; one explicit final-answer position',
                   'note':'Continuation of V2; its images/answers include predecessor development data. Rationale annotations preserve train partitions; no independent validation claim.'}
    args.output.mkdir(parents=True,exist_ok=True)
    config_path=args.output/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=configuration:raise RuntimeError('Training configuration changed on resume')
    if rank==0:
        temporary=config_path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(configuration,indent=2)+'\n');temporary.replace(config_path)
        (args.output/'selected_questions.json').write_text(json.dumps([r['id'] for r in records])+'\n')
    if (args.output/'complete.json').exists():dist.destroy_process_group();return
    saved=None
    if (args.output/'latest.json').exists():
        name=json.loads((args.output/'latest.json').read_text())['file']
        saved=torch.load(args.output/name,map_location='cpu',weights_only=False,mmap=True)
    initial=saved or torch.load(args.initialize,map_location='cpu',weights_only=False,mmap=True)
    model,wrapped=load_model('O2',42);stage(model,wrapped,'joint');restore_trainable(model,initial['adapter'])
    del initial
    if args.gradient_checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    opt=make_optimizer(model)
    global_step=0;start_epoch=1;start_begin=0
    if saved:
        opt.load_state_dict(saved['optimizer']);global_step=saved['global_step']
        start_epoch=saved['epoch']+int(saved['epoch_complete']);start_begin=0 if saved['epoch_complete'] else saved['next_begin']
        state=saved['rank_rng_states'][rank]
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);del saved
    else:
        # One experiment seed; rank-specific dropout streams within this run.
        torch.manual_seed(42+rank*100003);torch.cuda.manual_seed(42+rank*100003)
    ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    prefetch=FeaturePrefetch('O2',12);started=time.monotonic()
    planned=configuration['planned_steps'];warmup=max(1,math.ceil(planned*.03))

    def save(epoch,begin,complete=False):
        states=[None for _ in range(world)]
        dist.all_gather_object(states,rng_state(),group=control)
        if rank==0:
            name=f'epoch_{epoch}.pt' if complete else 'progress.pt'
            atomic_save({'adapter':trainable_state(model),'optimizer':opt.state_dict(),
                         'epoch':epoch,'epoch_complete':complete,'next_begin':begin,'global_step':global_step,
                         'rank_rng_states':states,'stage':'joint','configuration':configuration},args.output/name)
            temporary=args.output/'latest.tmp.json';temporary.write_text(json.dumps({'file':name})+'\n');temporary.replace(args.output/'latest.json')
        dist.barrier(group=control)

    try:
        for epoch in range(start_epoch,args.epochs+1):
            order=order_for_epoch(records,epoch);model.train();wrapped.visual.eval()
            if wrapped.alignment is not None:wrapped.alignment.train()
            wrapped.visual.merger.train();torch.cuda.reset_peak_memory_stats()
            for begin in range(start_begin if epoch==start_epoch else 0,len(order),32):
                group=[records[i] for i in order[begin:begin+32]]
                local_records=local_group(group,rank,world)
                if not local_records:raise RuntimeError('An empty rank requires explicit zero-weight handling')
                future=local_group([records[i] for i in order[begin+32:begin+64]],rank,world)
                futures=prefetch.submit(future)
                size=microbatch(max(r['input_ids'].numel() for r in group))
                factor=(global_step+1)/warmup if global_step<warmup else .1+.9*.5*(1+math.cos(math.pi*(global_step-warmup)/max(1,planned-warmup)))
                for g in opt.param_groups:g['lr']=g['initial_lr']*factor
                opt.zero_grad(set_to_none=True);stats=defaultdict(float)
                for i in range(0,len(local_records),size):
                    part=local_records[i:i+size]
                    sync=i+size>=len(local_records)
                    with nullcontext() if sync else ddp.no_sync():
                        batch,context=make_batch(part,'O2')
                        loss,values=objective(ddp,wrapped,batch,context,part)
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite rationale loss')
                        (loss*(world*len(part)/len(group))).backward()
                    for k,v in dict(loss=float(loss.detach()),**values).items():stats[k]+=v*len(part)/len(group)
                    del batch,context,loss
                for f in futures:f.result()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                opt.step();global_step+=1
                keys=sorted(stats);values=torch.tensor([stats[k] for k in keys],device='cuda',dtype=torch.float64);dist.all_reduce(values)
                if rank==0:
                    event=dict(epoch=epoch,global_step=global_step,planned_steps=planned,
                               completed_exposures=begin+len(group),microbatch=size,grad_norm=norm.item(),
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               elapsed_seconds=time.monotonic()-started,**dict(zip(keys,values.cpu().tolist())))
                    with (args.output/'training.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                    if global_step<=3 or global_step%10==0:print(json.dumps(event),flush=True)
                if global_step==1 or global_step%args.checkpoint_interval==0:save(epoch,begin+len(group))
                if args.max_steps and global_step>=args.max_steps:
                    save(epoch,begin+len(group));return
            save(epoch,len(order),True)
        if rank==0:
            (args.output/'complete.json').write_text(json.dumps({'global_steps':global_step,'epochs':args.epochs,
                'unique_questions':len(records),'seed':42,'final_checkpoint':f'epoch_{args.epochs}.pt',
                'elapsed_seconds_this_process':time.monotonic()-started,'completed_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    finally:
        prefetch.close();dist.destroy_process_group()


if __name__=='__main__':main()
