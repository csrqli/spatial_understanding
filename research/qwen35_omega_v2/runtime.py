"""V2 multi-image integration using the verified, unchanged V1 native backbone."""
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from research.qwen35_omega.runtime import (AlignedModel, add_lora, load_qwen, load_native_vision,
    load_spatial, prepare_spatial, processor, read_rows, trainable_state, restore_trainable)
from research.qwen35_omega.data import canonical_rgb
from research.qwen35_omega.fusion import masked_grid_pool
from .config import ARMS, WARM, RUN
from .fusion import Alignment


def load_model(arm,seed=42):
    manifest=json.loads((RUN/'splits/manifest.json').read_text())
    for reason in ('omni_train_test_overlap','omni_train_heldout_overlap'):
        if manifest['rejections'].get(reason,0):
            raise RuntimeError('Old warm-start saw newly excluded records; train a new common anchor before proceeding')
    model,_=load_qwen()
    wrapped=AlignedModel(model,arm)
    if arm!='B1':
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed+2000)
            fusion=Alignment(scene=arm=='O2').cuda()
        model.spatial_alignment=fusion
        wrapped.alignment=fusion
    add_lora(model,seed=42)
    wrapped.stage('B')
    saved=torch.load(WARM,map_location='cpu',weights_only=False,mmap=True)['adapter']
    expected={n for n,p in model.named_parameters() if p.requires_grad and not n.startswith('spatial_alignment.')}
    if expected!=set(saved):raise RuntimeError('Warm-start key mismatch')
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in saved:
                assert p.shape==saved[n].shape
                p.copy_(saved[n])
    del saved
    torch.manual_seed(seed+3000)
    return model,wrapped


def stage(model,wrapped,mode):
    if mode=='joint':wrapped.stage('B');return
    if mode!='alignment':raise ValueError(mode)
    model.requires_grad_(False).eval()
    if wrapped.alignment is not None:
        wrapped.alignment.requires_grad_(True).float().train()
    else:
        # Give the native baseline a useful, matched Stage-A pass as well.
        # Auxiliary arms preserve the common native path during this pass.
        wrapped.visual.merger.requires_grad_(True).float().train()


def collate(rows,p,training=True):
    texts,images,suffixes=[],[],[]
    for row in rows:
        content=[]
        for i,path in enumerate(row['images']):
            if len(row['images'])>1:content.append({'type':'text','text':f'Image {i+1}: '})
            content.append({'type':'image'})
            with Image.open(path) as im:images.append(canonical_rgb(im))
        content.append({'type':'text','text':row['prompt']})
        messages=[{'role':'user','content':content}]
        prefix=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        text=p.apply_chat_template(messages+[{'role':'assistant','content':row['target']}],tokenize=False,
                                   add_generation_prompt=False,enable_thinking=False) if training else prefix
        assert text.startswith(prefix)
        suffixes.append(p.tokenizer(text[len(prefix):],add_special_tokens=False).input_ids if training else [])
        texts.append(text)
    batch=dict(p(text=texts,images=images,padding=True,return_tensors='pt'))
    if batch['input_ids'].shape[1]>8192:raise ValueError('V2 sequence exceeds audited budget; no truncation')
    if training:
        labels=torch.full_like(batch['input_ids'],-100)
        for i,s in enumerate(suffixes):
            assert s and batch['input_ids'][i,-len(s):].tolist()==s
            labels[i,-len(s):]=torch.tensor(s)
        batch['labels']=labels
    return batch


@torch.no_grad()
def spatial_batch(encoder,kind,rows):
    counts={len(r['images']) for r in rows}
    if len(counts)!=1:raise ValueError('Spatial encoder batches require equal frame counts')
    T=next(iter(counts));B=len(rows)
    prep=[prepare_spatial({'image':p},kind) for r in rows for p in r['images']]
    images=torch.stack([x[0] for x in prep]).cuda().reshape(B,T,*prep[0][0].shape)
    weights=torch.stack([x[1] for x in prep]).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):features,start=encoder(images)
    final=features[-1].reshape(B*T,features[-1].shape[-2],2048)
    side,patch=(416,16) if kind=='omega' else (518,14)
    assert start==(17 if kind=='omega' else 5)
    grid=final[:,start:].reshape(B*T,side//patch,side//patch,2048)
    local,valid=masked_grid_pool(grid.float(),weights,14)
    pooled,pvalid=masked_grid_pool(grid.float(),weights,4)
    camera=final[:,:1]
    values={'local':torch.cat((local,camera),1).bfloat16(),
            'local_valid':torch.cat((valid,torch.ones(B*T,1,device='cuda',dtype=torch.bool)),1),
            'pooled_scene':torch.cat((camera,pooled),1).bfloat16(),
            'pooled_valid':torch.cat((torch.ones(B*T,1,device='cuda',dtype=torch.bool),pvalid),1)}
    if kind=='omega':values['register_scene']=final[:,:17].bfloat16().contiguous()
    return [{k:v[i*T:(i+1)*T].cpu().contiguous() for k,v in values.items()} for i in range(B)]


def objective(model,wrapped,batch,context,answer_ids):
    batch={k:v.cuda() for k,v in batch.items()}
    labels=batch.pop('labels')
    shifted=F.pad(labels,(0,1),value=-100)[:,1:]
    indices=torch.where((shifted!=-100).any(0))[0]
    target=shifted[:,indices]
    valid=target!=-100
    answer=torch.isin(target,torch.tensor(answer_ids,device='cuda'))&valid
    assert int(answer.sum())==labels.shape[0]
    wrapped.context=context
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(**batch,logits_to_keep=indices,use_cache=False).logits.float()
            ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').view_as(target)
            a=ce[answer].mean();f=ce[valid&~answer].mean()
            loss=.5*a+.5*f
        stats={'answer_ce':a.detach().item(),'format_ce':f.detach().item(),
               'answer_accuracy':(logits.argmax(-1)[answer]==target[answer]).float().mean().detach().item()}
        return loss,stats
    finally:wrapped.context=None
