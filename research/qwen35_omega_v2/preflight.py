"""Multi-image correctness, identity preservation and real optimizer-step profiles."""
from collections import defaultdict
import gc
import json
import time
import torch
from .config import CACHE,RUN
from .cache import make_batch
from .runtime import load_model,collate,processor,objective,stage,trainable_state,restore_trainable,read_rows
from .train import microbatch,optimizer


def main():
    folder=RUN/'profiles/preflight';folder.mkdir(parents=True,exist_ok=True)
    tokens=torch.load(CACHE/'tokens.pt',map_location='cpu',weights_only=True)
    rows=read_rows(RUN/'splits/tiny_train.jsonl')
    sample=[next(r for r in rows if r['source']==s) for s in ('omni','mind','sat')]
    records=[tokens[r['id']] for r in sample]
    letters=json.loads((CACHE/'tokens_full_train_0_complete.json').read_text())['answer_token_ids']
    model,wrapped=load_model('O2',42)
    stage(model,wrapped,'alignment');model.eval()
    batch,context=make_batch(records,'O2')
    p=processor();online=collate(sample,p)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        reference=wrapped.original_forward(online['pixel_values'].cuda().bfloat16(),grid_thw=online['image_grid_thw'].cuda()).last_hidden_state
        relative=((reference.float()-context['native'].float()).norm()/reference.float().norm()).item()
        assert relative<.01,relative
        expected=wrapped.visual.merger(context['native'])
        wrapped.context=context
        actual=wrapped.visual(torch.empty(0,device='cuda'),grid_thw=context['grid'])
        wrapped.context=None
        assert actual.last_hidden_state.dtype==context['native'].dtype
        torch.testing.assert_close(actual.last_hidden_state,context['native'],rtol=0,atol=0)
        torch.testing.assert_close(actual.pooler_output,expected,rtol=0,atol=0)
    report={'passed':False,'gpu':torch.cuda.get_device_name(),'native_cache_online_relative_l2':relative,
            'zero_adapter_native_and_merger_exact_identity':True,'multi_image_order':[len(r['images']) for r in sample]}
    # Choice scoring must not depend on the stored correct answer token.
    altered=[dict(r,input_ids=r['input_ids'].clone()) for r in records]
    for row in altered:row['input_ids'][row['answer_position']]=letters['D'] if row['answer']!='D' else letters['A']
    first,_=make_batch(records,'O2','choice');second,_=make_batch(altered,'O2','choice')
    for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    report['choice_inputs_independent_of_true_letter']=True
    del reference,expected,actual,online,first,second
    opt=optimizer(model,'alignment',1e-4)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss,_=objective(model,wrapped,batch,context,list(letters.values()));loss.backward()
        assert torch.isfinite(loss)
        for name in ('local_output.weight','scene_output.weight'):
            grad=dict(wrapped.alignment.named_parameters())[name].grad
            assert grad is not None and grad.abs().max()>0,name
        if step==1:
            for name in ('local_input.weight','scene_input.weight'):
                assert dict(wrapped.alignment.named_parameters())[name].grad.abs().max()>0,name
        assert all(q.grad is None for n,q in model.named_parameters() if not n.startswith('spatial_alignment.'))
        opt.step()
    report['zero_head_then_upstream_gradients']=True
    model.zero_grad(set_to_none=True);model.eval()
    state=trainable_state(model)
    with torch.no_grad():before=objective(model,wrapped,batch,context,list(letters.values()))[0].item()
    with torch.no_grad():wrapped.alignment.local_gate.add_(.3)
    restore_trainable(model,state)
    with torch.no_grad():after=objective(model,wrapped,batch,context,list(letters.values()))[0].item()
    assert before==after
    report['strict_adapter_restore']=True
    del state,opt,batch,context
    # Stress actual longest records in every populated batching bucket, including
    # optimizer moments. This probe is not a scored experiment/checkpoint.
    all_rows=read_rows(RUN/'splits/full_train.jsonl')
    grouped=defaultdict(list)
    for row in all_rows:
        record=tokens[row['id']]
        length=((record['input_ids'].numel()+127)//128)*128
        grouped[microbatch(length)].append(record)
    stage(model,wrapped,'joint')
    opt=optimizer(model,'joint',1e-4)
    profiles=[]
    for size,values in sorted(grouped.items(),reverse=True):
        longest=max(values,key=lambda r:r['input_ids'].numel())
        part=[longest]*size
        opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        batch,context=make_batch(part,'O2')
        durations=[]
        for _ in range(2):
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
            loss,stats=objective(model,wrapped,batch,context,list(letters.values()))
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize();durations.append(time.monotonic()-start)
        item={'microbatch':size,'sequence_bucket':batch['input_ids'].shape[1],
              'images_per_question':len(longest['image_hashes']),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'step_seconds':durations,'loss':loss.item()}
        assert item['peak_allocated_gib']<240,item
        profiles.append(item);print(json.dumps(item),flush=True)
        del batch,context,loss
    report.update(profiles=profiles,passed=True)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
