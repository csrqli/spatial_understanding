"""Baseline-anchored alignment, then matched joint adaptation with audit logs."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from research.qwen35_omega.cache import atomic_save
from .config import CACHE,RUN,ARMS,WARM,TRAINING_SEEDS
from .cache import make_batch,identity,native_path
from .runtime import load_model,stage,objective,read_rows,trainable_state,restore_trainable


def microbatch(length):
    return 32 if length<=768 else 16 if length<=1536 else 8 if length<=3072 else 4 if length<=6144 else 2


def kind(name):
    return 'fusion' if name.startswith('spatial_alignment.') else 'lora' if 'lora_' in name else 'merger'


def optimizer(model,mode,adapter_lr):
    rates={'fusion':adapter_lr if mode=='alignment' else adapter_lr/2,'merger':3e-6,'lora':1e-5}
    grouped={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        component=kind(name)
        decay=p.ndim>1 and not any(k in name for k in ('norm','gate','bias'))
        key=(component,decay)
        if key not in grouped:grouped[key]={'params':[],'lr':rates[component],'initial_lr':rates[component],
                                           'weight_decay':.01 if decay else 0.,'component':component}
        grouped[key]['params'].append(p)
    return torch.optim.AdamW(list(grouped.values()),betas=(.9,.95),eps=1e-8)


def epoch_order(records,seed,epoch,replay=True):
    rng=random.Random(seed+epoch*1009)
    repetitions={'sat':1,'omni':4,'mind':2} if replay else {'sat':1,'omni':1,'mind':1}
    order=[i for i,r in enumerate(records) for _ in range(repetitions[r['source']])]
    rng.shuffle(order)
    for begin in range(0,len(order),512):order[begin:begin+512]=sorted(order[begin:begin+512],key=lambda i:records[i]['input_ids'].numel())
    return order


def gradients(model):
    components=defaultdict(list)
    for name,p in model.named_parameters():
        if p.requires_grad and p.grad is not None:components[kind(name)].append(p.grad.detach())
    return {k:torch.linalg.vector_norm(torch.stack([g.float().norm() for g in v])).item() for k,v in components.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS),required=True)
    p.add_argument('--seed',type=int,choices=TRAINING_SEEDS,default=TRAINING_SEEDS[0]);p.add_argument('--subset',default='pilot_train')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--joint-epochs',type=int,default=3)
    p.add_argument('--adapter-lr',type=float,default=1e-4);p.add_argument('--tiny-steps',type=int,default=0)
    p.add_argument('--gradient-checkpointing',action='store_true');args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    all_tokens=torch.load(CACHE/'tokens.pt',map_location='cpu',weights_only=True)
    rows=read_rows(RUN/'splits'/f'{args.subset}.jsonl')
    records=[all_tokens[r['id']] for r in rows];del all_tokens
    if not all(native_path(h).exists() for h in {h for r in rows for h in r['image_hashes']}):raise RuntimeError('Missing native cache')
    if ARMS[args.arm] and not all((CACHE/ARMS[args.arm]/f'{k}.pt').exists() for k in {r['feature_key'] for r in rows}):raise RuntimeError('Missing spatial cache')
    answer_ids=list(json.loads((CACHE/'tokens_full_train_0_complete.json').read_text())['answer_token_ids'].values())
    model,wrapped=load_model(args.arm,args.seed)
    if args.gradient_checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    args.output.mkdir(parents=True,exist_ok=True)
    configuration={**vars(args),'output':str(args.output),'records':len(records),'identity':identity(),
                   'split_sha256':hashlib.sha256((RUN/'splits'/f'{args.subset}.jsonl').read_bytes()).hexdigest(),
                   'warm_checkpoint':str(WARM),'warm_size':WARM.stat().st_size,'warm_mtime_ns':WARM.stat().st_mtime_ns,
                   'code_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                  for n in ('train.py','runtime.py','fusion.py','cache.py')}}
    config=args.output/'configuration.json'
    if config.exists() and json.loads(config.read_text())!=configuration:raise RuntimeError('Incompatible run resume')
    config.write_text(json.dumps(configuration,indent=2)+'\n')
    if (args.output/'complete.json').exists():print('Already complete');return
    # Save actual shared initialization fingerprints for all paired arms.
    init={k:hashlib.sha256() for k in ('merger','lora','fusion')}
    for n,value in model.named_parameters():
        if n.startswith('spatial_alignment.') or 'lora_' in n or '.visual.merger.' in n:
            init[kind(n)].update(n.encode());init[kind(n)].update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    (args.output/'initialization.json').write_text(json.dumps({k:v.hexdigest() for k,v in init.items()},indent=2)+'\n')
    resume=None
    if (args.output/'latest.json').exists():
        name=json.loads((args.output/'latest.json').read_text())['file']
        resume=torch.load(args.output/name,map_location='cpu',weights_only=False)
    started=time.monotonic();global_step=resume['global_step'] if resume else 0
    def save(epoch,mode,opt,next_begin,stage_step,epoch_complete=False):
        value={'adapter':trainable_state(model),'optimizer':opt.state_dict() if opt else None,'epoch':epoch,'stage':mode,
               'next_begin':next_begin,'stage_step':stage_step,'epoch_complete':epoch_complete,'global_step':global_step,
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
               'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'configuration':configuration}
        name=f'epoch_{epoch}.pt' if epoch_complete else 'progress.pt'
        atomic_save(value,args.output/name)
        temp=args.output/'latest.tmp.json';temp.write_text(json.dumps({'file':name})+'\n');os.replace(temp,args.output/'latest.json')
    joint_start=1 if args.tiny_steps else 2
    schedule=[(e,'joint' if e>=joint_start else 'alignment') for e in range(1,(1 if args.tiny_steps else args.joint_epochs+1)+1)]
    opt=None;previous_mode=None;stage_step=0
    for epoch,mode in schedule:
        if resume and (epoch<resume['epoch'] or epoch==resume['epoch'] and resume['epoch_complete']):continue
        stage(model,wrapped,mode)
        order=epoch_order(records,args.seed,epoch,replay=not args.tiny_steps)
        if args.tiny_steps:order=(order*math.ceil(args.tiny_steps*32/len(order)))[:args.tiny_steps*32]
        if mode!=previous_mode:
            opt=optimizer(model,mode,args.adapter_lr)
            stage_step=0;previous_mode=mode
        begin_at=0
        if resume:
            stage(model,wrapped,resume['stage']);restore_trainable(model,resume['adapter']);stage(model,wrapped,mode)
            if resume['stage']==mode and opt:
                opt.load_state_dict(resume['optimizer']);stage_step=resume['stage_step']
            if epoch==resume['epoch'] and not resume['epoch_complete']:begin_at=resume['next_begin']
            torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
            random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);resume=None
        if opt is None:
            save(epoch,mode,None,len(order),0,True)
            print('Baseline alignment stage is an exact no-op: common native path stays frozen.',flush=True)
            continue
        steps=math.ceil(len(order)/32)
        total=steps*(1 if mode=='alignment' or args.tiny_steps else args.joint_epochs)
        warmup=max(1,math.ceil(total*.03))
        torch.cuda.reset_peak_memory_stats()
        for begin in range(begin_at,len(order),32):
            group=[records[i] for i in order[begin:begin+32]]
            bucket=math.ceil(max(r['input_ids'].numel() for r in group)/128)*128
            size=microbatch(bucket)
            opt.zero_grad(set_to_none=True)
            factor=(stage_step+1)/warmup if stage_step<warmup else .5*(1+math.cos(math.pi*(stage_step-warmup)/max(1,total-warmup)))
            for g in opt.param_groups:g['lr']=g['initial_lr']*factor
            stats=defaultdict(float)
            for i in range(0,len(group),size):
                part=group[i:i+size];batch,context=make_batch(part,args.arm)
                loss,values=objective(model,wrapped,batch,context,answer_ids)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss*len(part)/len(group)).backward()
                for k,v in dict(loss=loss.item(),**values).items():stats[k]+=v*len(part)/len(group)
                del loss,batch,context
            per_group=gradients(model)
            norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
            opt.step();stage_step+=1;global_step+=1
            event=dict(epoch=epoch,stage=mode,global_step=global_step,examples=begin+len(group),
                       microbatch=size,bucket=bucket,grad_norm=norm.item(),group_grad_norms=per_group,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       elapsed_seconds=time.monotonic()-started,**stats)
            if wrapped.alignment is not None:event.update(wrapped.alignment.last_diagnostics)
            with (args.output/'training.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            if global_step%10==0:print(json.dumps(event),flush=True)
            if global_step%500==0:save(epoch,mode,opt,begin+len(group),stage_step)
        save(epoch,mode,opt,len(order),stage_step,True)
        print('epoch complete',epoch,flush=True)
    (args.output/'complete.json').write_text(json.dumps({'epochs':len(schedule),'global_steps':global_step,
        'unique_records':len(records),'joint_epochs':args.joint_epochs,'tiny_steps':args.tiny_steps,
        'elapsed_seconds':time.monotonic()-started,'final_checkpoint':f'epoch_{schedule[-1][0]}.pt'},indent=2)+'\n')


if __name__=='__main__':main()
