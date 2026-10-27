"""Frozen caches: native per image, spatial per ordered multi-view sequence."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import torch
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from research.qwen35_omega.cache import atomic_save, NativeCollator, identity as old_identity
from .config import CACHE,RUN,V1,ARMS
from .runtime import read_rows,processor,collate,load_native_vision,load_spatial,spatial_batch


@lru_cache(maxsize=1)
def identity():
    return {'schema':'q35v2_joint_multiview_features','native':'qwen35-9b-last-premerger-bf16',
            'qwen_revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
            'omega_sha256':'e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896',
            'vggt_sha256':'f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e',
            'qwen_pixels':[65536,1003520],'auxiliary':'white-square-no-crop-omega416-vggt518',
            'auxiliary_context':'joint encoder forward over all ordered images in a record',
            'extraction_code_sha256':hashlib.sha256((inspect.getsource(spatial_batch)+inspect.getsource(load_native_vision)).encode()).hexdigest()}


def native_path(digest):
    path=CACHE/'native'/f'{digest}.pt'
    if path.exists():return path
    old=V1/'cache_v1/native'/f'{digest}.pt'
    return old if old.exists() else path


# Full refits touch 50,933 native images and at most 35,843 ordered spatial
# sequences. Keeping those CPU-side tensors hot avoids per-step random NFS
# reads; the packed cache footprint is about 153 GiB for native plus one
# spatial source, so full refits request 224 GiB of host memory.
@lru_cache(maxsize=60_000)
def native(digest):
    path=native_path(digest)
    item=torch.load(path,map_location='cpu',weights_only=True)
    assert item['image_sha256']==digest
    if V1 in path.parents:
        assert item['cache_identity']==old_identity()
    else:assert item['identity']==identity()
    return item


@lru_cache(maxsize=40_000)
def auxiliary(kind,key):
    item=torch.load(CACHE/kind/f'{key}.pt',map_location='cpu',weights_only=True)
    assert item['feature_key']==key and item['identity']==identity()
    return item


def make_batch(records,arm,mode='train'):
    ids,grids,natives,aux=[],[],[],[]
    for row in records:
        end=row['input_ids'].numel() if mode=='train' else row['answer_position'] if mode=='choice' else row['prompt_length']
        ids.append(row['input_ids'][:end].long())
        values=[native(h) for h in row['image_hashes']]
        grid=torch.cat([v['grid'] for v in values]).long()
        assert torch.equal(grid,row['grid'].long())
        grids.append(grid);natives.extend(v['native'] for v in values)
        if ARMS[arm]:aux.append(auxiliary(ARMS[arm],row['feature_key']))
    length=math.ceil(max(t.numel() for t in ids)/128)*128
    def pad(v,fill):return torch.stack([torch.nn.functional.pad(x,(length-x.numel(),0),value=fill) for x in v])
    padded=pad(ids,0)
    image_id=records[0]['image_token_id']
    batch={'input_ids':padded,'attention_mask':pad([torch.ones_like(x) for x in ids],0),
           'mm_token_type_ids':(padded==image_id).long(),'image_grid_thw':torch.cat(grids),'pixel_values':torch.empty(0)}
    if mode=='train':
        labels=torch.full_like(padded,-100)
        for i,row in enumerate(records):labels[i,-row['suffix_length']:]=padded[i,-row['suffix_length']:]
        batch['labels']=labels
    context={'native':torch.cat(natives).cuda(non_blocking=True),'grid':batch['image_grid_thw'].cuda()}
    if aux:
        for key in ('local','local_valid','pooled_scene','pooled_valid','register_scene'):
            if key in aux[0]:context[key]=torch.cat([a[key] for a in aux]).cuda(non_blocking=True)
    return batch,context


def fast_tokenize(row,p,metadata):
    grids=[]
    for digest in row['image_hashes']:
        m=metadata[digest]
        h,w=smart_resize(m['height'],m['width'],factor=32,min_pixels=65536,max_pixels=1003520)
        grids.append([1,h//16,w//16])
    grid=torch.tensor(grids,dtype=torch.long)
    content=[]
    for i in range(len(grids)):
        if len(grids)>1:content.append({'type':'text','text':f'Image {i+1}: '})
        content.append({'type':'image'})
    content.append({'type':'text','text':row['prompt']})
    messages=[{'role':'user','content':content}]
    prefix=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    text=p.apply_chat_template(messages+[{'role':'assistant','content':row['target']}],tokenize=False,
                               add_generation_prompt=False,enable_thinking=False)
    assert text.startswith(prefix)
    replacements=[p.replace_image_token({'image_grid_thw':grid},i) for i in range(len(grids))]
    expanded,_=p.get_text_with_replacements([text],replacements,[],[])
    ids=p.tokenizer(expanded[0],add_special_tokens=False).input_ids
    suffix=p.tokenizer(text[len(prefix):],add_special_tokens=False).input_ids
    assert ids[-len(suffix):]==suffix and len(suffix)==5
    answer_position=len(ids)-3
    assert p.tokenizer.decode([ids[answer_position]]).strip()==row['answer']
    if len(ids)>8192:raise ValueError(f"Sequence too long; no silent truncation: {row['id']} {len(ids)}")
    return {**{k:row[k] for k in ('id','image_hashes','feature_key','source','task_type','sub_task_type','answer','choices')},
            'input_ids':torch.tensor(ids,dtype=torch.int32),'grid':grid.to(torch.int32),
            'suffix_length':len(suffix),'prompt_length':len(ids)-len(suffix),'answer_position':answer_position,
            'image_token_id':p.image_token_ids[0]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=('tokens','native','omega','vggt'),required=True)
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--subset',default='full_train');parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args();CACHE.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    rows=read_rows(RUN/'splits'/f'{args.subset}.jsonl')
    if len({row['id'] for row in rows})!=len(rows):raise RuntimeError('Duplicate question IDs would overwrite cached tokens')
    if args.kind=='tokens':
        target=CACHE/('test_tokens.pt' if args.subset=='test' else 'tokens.pt')
        if target.exists():raise FileExistsError(target)
        catalog='test_images.jsonl' if args.subset=='test' else 'images.jsonl'
        p=processor();meta={r['image_sha256']:r for r in read_rows(RUN/'splits'/catalog)}
        tokens={}
        for i,row in enumerate(rows):
            record=fast_tokenize(row,p,meta)
            # Verify exact HF processor/template parity on diverse real examples.
            if i<4 or i%(max(1,len(rows)//32))==0:
                reference=collate([row],p)
                assert torch.equal(reference['input_ids'][0],record['input_ids'].long())
                assert torch.equal(reference['image_grid_thw'],record['grid'].long())
                assert torch.equal(reference['mm_token_type_ids'][0],(record['input_ids']==record['image_token_id']).long())
            tokens[row['id']]=record
            if (i+1)%2000==0:print('tokens',i+1,'/',len(rows),flush=True)
        if len(tokens)!=len(rows):raise RuntimeError('Token cache does not cover every question')
        atomic_save(tokens,target)
        letters={a:p.tokenizer.encode('Answer: '+a,add_special_tokens=False)[-1] for a in 'ABCD'}
        report={'count':len(tokens),'max_sequence_length':max(r['input_ids'].numel() for r in tokens.values()),
                'answer_token_ids':letters,'fast_processor_exact_parity':True}
    elif args.kind=='native':
        required={h for r in rows for h in r['image_hashes']}
        catalog='test_images.jsonl' if args.subset=='test' else 'images.jsonl'
        images=[r for r in read_rows(RUN/'splits'/catalog) if r['image_sha256'] in required]
        selected=sorted(images,key=lambda r:r['image_sha256'])[args.shard::args.shards]
        pending=[r for r in selected if not native_path(r['image_sha256']).exists()]
        model=load_native_vision();p=processor()
        loader=torch.utils.data.DataLoader(pending,batch_size=args.batch_size,num_workers=8,pin_memory=True,
                                           collate_fn=NativeCollator(p),prefetch_factor=2)
        for begin,batch in zip(range(0,len(pending),args.batch_size),loader):
            grids=batch['image_grid_thw'].cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                features=model(batch['pixel_values'].cuda().bfloat16(),grid_thw=grids).last_hidden_state
            for row,feat,grid in zip(pending[begin:begin+args.batch_size],features.split(grids.prod(-1).tolist()),grids):
                atomic_save({'native':feat.cpu().contiguous(),'grid':grid.cpu()[None],'image_sha256':row['image_sha256'],
                             'identity':identity()},CACHE/'native'/f"{row['image_sha256']}.pt")
            if begin%(args.batch_size*10)==0:print('native',args.shard,begin+len(grids),'/',len(pending),flush=True)
        assert all(native_path(r['image_sha256']).exists() for r in selected)
        report={'count':len(selected),'new':len(pending),'reused':len(selected)-len(pending)}
    else:
        unique={r['feature_key']:r for r in rows}
        selected=[unique[k] for k in sorted(unique)][args.shard::args.shards]
        pending=[r for r in selected if not (CACHE/args.kind/f"{r['feature_key']}.pt").exists()]
        by_frames=defaultdict(list)
        for r in pending:by_frames[len(r['images'])].append(r)
        model=load_spatial(args.kind)
        done=0
        for frames,group in sorted(by_frames.items()):
            # Bound quadratic cross-view work and total patches on B300.
            size=max(1,min(args.batch_size,128//frames**2))
            for begin in range(0,len(group),size):
                part=group[begin:begin+size]
                values=spatial_batch(model,args.kind,part)
                if begin==0:
                    single=spatial_batch(model,args.kind,part[:1])[0]
                    for key,value in single.items():
                        if value.dtype==torch.bool:assert torch.equal(value,values[0][key])
                        else:
                            error=(value.float()-values[0][key].float()).norm()/value.float().norm().clamp_min(1e-8)
                            assert float(error)<.05,(args.kind,frames,key,float(error))
                for row,value in zip(part,values):
                    value.update(feature_key=row['feature_key'],image_hashes=row['image_hashes'],identity=identity())
                    atomic_save(value,CACHE/args.kind/f"{row['feature_key']}.pt")
                done+=len(part)
                if begin%(size*10)==0:print(args.kind,args.shard,done,'/',len(pending),'frames',frames,flush=True)
        assert all((CACHE/args.kind/f"{r['feature_key']}.pt").exists() for r in selected)
        report={'count':len(selected),'new':len(pending)}
    report.update(kind=args.kind,shard=args.shard,shards=args.shards,subset=args.subset,
                  elapsed_seconds=time.monotonic()-start,identity=identity())
    if args.kind!='tokens':report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,gpu=torch.cuda.get_device_name())
    path=CACHE/f'{args.kind}_{args.subset}_{args.shard}_complete.json'
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
