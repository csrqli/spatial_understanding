"""Matched native/aligned generation evaluation, reusing verified frozen caches.

Launch two independent replicas with torchrun: rank 0 is the pretrained native
model, rank 1 is the completed O2 checkpoint. No gradients or test-time fitting.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .cache import make_batch
from .config import CACHE, FORMAT, ROOT, RUN
from .evaluate import score, summarize, validate_records
from .runtime import (AlignedModel, load_model, load_qwen, processor,
                      read_rows, restore_trainable, stage)

SYSTEM = '''You are a spatial-reasoning assistant.
Ground every conclusion in the supplied image. Identify the relevant objects,
reference frame, viewpoint, and spatial relations before solving the question.
For motion, track the direction and time ordering. For geometry or patterns,
work through the transformations and check the candidate options.
Reason carefully, then end your response with a separate line: Answer: X
where X is the selected option letter. Choose the most plausible valid option.'''


def parse_answer(response, choices):
    final = response.rsplit('</think>', 1)[-1].replace('<|im_end|>','')
    matches = re.findall(r'\bAnswer\s*:\s*([A-D])\b', final, flags=re.I)
    if matches:
        answer = matches[-1].upper()
        return answer if answer in choices else None
    matches = re.findall(r'\\boxed\{\s*([A-D])\s*\}', final, flags=re.I)
    if matches:
        answer = matches[-1].upper()
        return answer if answer in choices else None
    final = final.strip()
    return final.upper() if len(final) == 1 and final.upper() in choices else None


def generation_record(row, old, p, thinking=True):
    # Remove only the original shared instruction; preserve question/options.
    if not row['prompt'].endswith(FORMAT):
        raise RuntimeError(f"Unknown prompt format: {row['id']}")
    question = row['prompt'][:-len(FORMAT)].rstrip()
    question += '\n\nSelect one valid option. End with Answer: X.'
    content = []
    for i in range(len(row['image_hashes'])):
        if len(row['image_hashes']) > 1:
            content.append({'type':'text','text':f'Image {i+1}: '})
        content.append({'type':'image'})
    content.append({'type':'text','text':question})
    messages = [{'role':'system','content':SYSTEM}, {'role':'user','content':content}]
    prefix = p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)
    replacements = [p.replace_image_token({'image_grid_thw':old['grid'].long()},i)
                    for i in range(len(row['image_hashes']))]
    expanded,_ = p.get_text_with_replacements([prefix],replacements,[],[])
    ids = p.tokenizer(expanded[0],add_special_tokens=False).input_ids
    if len(ids)>8192:
        raise RuntimeError(f"Generation prompt exceeds audited budget: {row['id']}")
    return dict(old,input_ids=torch.tensor(ids,dtype=torch.int32),suffix_length=0,
                prompt_length=len(ids),answer_position=len(ids))


def decision_record(record, generated_ids, p):
    """Preserve reasoning, close thinking, and request one option-letter logit.

    The same readout applies to every item, independent of correctness or
    whether natural generation already produced an answer. No ground truth
    enters this input construction.
    """
    closing=p.tokenizer.convert_tokens_to_ids('</think>')
    reasoning=[]
    for token in generated_ids:
        if token in (closing,p.tokenizer.eos_token_id):break
        reasoning.append(token)
    suffix=p.tokenizer.encode('\n</think>\n\nAnswer:',add_special_tokens=False)
    ids=torch.cat((record['input_ids'],torch.tensor(reasoning+suffix,dtype=torch.int32)))
    return dict(record,input_ids=ids,prompt_length=len(ids),answer_position=len(ids))


class PresencePenalty(LogitsProcessor):
    """Apply Qwen's recommended penalty to generated tokens only."""
    def __init__(self,start,penalty):
        self.start,self.penalty = start,penalty

    def __call__(self, input_ids, scores):
        if input_ids.shape[1]>self.start:
            present=torch.zeros_like(scores)
            present.scatter_(1,input_ids[:,self.start:],1)
            scores=scores-present*self.penalty
        return scores


def write_json(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(path)


def score_native_durable(model,wrapped,records,rows,letters,output):
    path=output/'predictions.jsonl'
    predictions=read_rows(path) if path.exists() else []
    ids={r['id'] for r in predictions}
    expected={r['id'] for r in rows}
    if len(ids)!=len(predictions) or not ids.issubset(expected):
        raise RuntimeError('Native choice checkpoint contains unexpected IDs')
    pending=[(r,row) for r,row in zip(records,rows) if row['id'] not in ids]
    for begin in range(0,len(pending),32):
        part=pending[begin:begin+32]
        values=score(model,wrapped,[r for r,row in part],[row for r,row in part],
                     'B1',letters,8)
        with path.open('a') as f:
            for value in values:f.write(json.dumps(value)+'\n')
            f.flush();os.fsync(f.fileno())
        predictions.extend(values)
        write_json(output/'progress.json',{'completed':len(predictions),'expected':len(rows),
                   'updated_utc':datetime.now(timezone.utc).isoformat()})
        print('native_choice_checkpoint',len(predictions),'/',len(rows),flush=True)
    assert {r['id'] for r in predictions}==expected
    return predictions


@torch.inference_mode()
def generate(model,wrapped,records,rows,p,args,output):
    protocol = dict(thinking=not args.no_thinking, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,top_p=.95,top_k=20,presence_penalty=args.presence_penalty,
                    seed=42,batch_size=args.batch_size,system_prompt=SYSTEM,
                    input_protocol='No answer-letter prefix; no ground truth in input.',
                    scoring='Last Answer: X in final response; boxed or sole-letter fallback; invalid counts wrong.')
    if args.reason_then_choice:
        protocol['scoring']='All items: reasoning followed by fixed </think> Answer: prefix and argmax over valid option-letter logits. Natural-generation accuracy reported separately.'
    manifest=dict(arm=wrapped.arm,checkpoint=str(args.checkpoint) if wrapped.arm=='O2' else None,
                  ids=[r['id'] for r in rows],protocol=protocol,
                  scope='Engineering smoke on training-side examples' if args.smoke else 'Full official OmniSpatial test',
                  expected_n=len(rows),created_utc=datetime.now(timezone.utc).isoformat())
    manifest['input_sha256']=hashlib.sha256(b''.join(r['input_ids'].numpy().tobytes() for r in records)).hexdigest()
    lock=output/'manifest.json'
    if lock.exists():
        prior=json.loads(lock.read_text())
        for key in ['arm','checkpoint','ids','protocol','input_sha256']:
            if prior[key]!=manifest[key]:raise RuntimeError(f'Resume manifest mismatch: {key}')
    else:write_json(lock,manifest)
    path=output/'predictions.jsonl'
    previous=read_rows(path) if path.exists() else []
    seen={r['id'] for r in previous}
    if len(seen)!=len(previous) or not seen.issubset(set(manifest['ids'])):
        raise RuntimeError('Unexpected or duplicate resumed predictions')
    pending=[(r,row) for r,row in zip(records,rows) if row['id'] not in seen]
    # Stable order and fixed seed; no second rollout or answer-based retries.
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    if previous and pending:
        rng=torch.load(output/'rng.pt',map_location='cpu',weights_only=True)
        torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'])
    start=time.monotonic();model.eval()
    for begin in range(0,len(pending),args.batch_size):
        part=pending[begin:begin+args.batch_size]
        batch,context=make_batch([r for r,row in part],wrapped.arm,mode='generate')
        assert 'labels' not in batch
        length=batch['input_ids'].shape[1]
        wrapped.context=context
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                generated=model.generate(**{k:v.cuda() for k,v in batch.items()},
                    max_new_tokens=args.max_new_tokens,do_sample=args.temperature>0,
                    temperature=args.temperature if args.temperature>0 else None,
                    top_p=.95 if args.temperature>0 else None,top_k=20 if args.temperature>0 else None,
                    logits_processor=LogitsProcessorList([PresencePenalty(length,args.presence_penalty)]),
                    pad_token_id=p.tokenizer.pad_token_id,eos_token_id=p.tokenizer.eos_token_id,
                    use_cache=True)
        finally:wrapped.context=None
        decisions=None
        if args.reason_then_choice:
            readouts=[decision_record(record,ids.tolist(),p)
                      for (record,row),ids in zip(part,generated[:,length:].cpu())]
            readout_batch,readout_context=make_batch(readouts,wrapped.arm,mode='generate')
            wrapped.context=readout_context
            try:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(**{k:v.cuda() for k,v in readout_batch.items()},
                                 logits_to_keep=1,use_cache=False).logits[:,-1].float()
                decisions=[]
                for i,(record,row) in enumerate(part):
                    letters=[p.tokenizer.encode('Answer: '+a,add_special_tokens=False)[-1]
                             for a in row['choices']]
                    values=logits[i,torch.tensor(letters,device='cuda')]
                    decisions.append((row['choices'][int(values.argmax())],values.cpu().tolist()))
            finally:wrapped.context=None
            del readout_batch,readout_context,logits
        predictions=[]
        for i,((record,row),tokens) in enumerate(zip(part,generated[:,length:])):
            ids=tokens.tolist();eos=p.tokenizer.eos_token_id
            ended=eos in ids
            if ended:ids=ids[:ids.index(eos)+1]
            response=p.tokenizer.decode(ids,skip_special_tokens=False)
            answer=parse_answer(response,row['choices'])
            result={k:row[k] for k in ['id','source','task_type','sub_task_type','group_id','answer']}
            result.update(predicted=answer,correct=answer==row['answer'],response=response,
                          generated_tokens=len(ids),truncated=not ended and len(ids)>=args.max_new_tokens,
                          completed_thinking='</think>' in response)
            if decisions is not None:
                selected,values=decisions[i]
                result.update(generated_answer=answer,generated_correct=answer==row['answer'],
                              predicted=selected,correct=selected==row['answer'],choice_logits=values,
                              answer_readout='reason_then_choice')
            predictions.append(result)
        # One durable prediction per item. An interruption never changes a
        # completed answer; RNG is recorded for continuation after each batch.
        with path.open('a') as f:
            for result in predictions:f.write(json.dumps(result)+'\n')
            f.flush();os.fsync(f.fileno())
        previous.extend(predictions)
        torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()},output/'rng.pt')
        progress={'completed':len(previous),'expected':len(rows),
                  'elapsed_seconds_this_process':time.monotonic()-start,
                  'tokens_this_process':sum(x['generated_tokens'] for x in previous[-(begin+len(part)):]),
                  'truncated':sum(x['truncated'] for x in previous),
                  'invalid_answers':sum(x['predicted'] is None for x in previous),
                  'updated_utc':datetime.now(timezone.utc).isoformat(),
                  'gpu':torch.cuda.get_device_name(),
                  'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        write_json(output/'progress.json',progress)
        print(json.dumps(progress),flush=True)
        del batch,context,generated
    if {r['id'] for r in previous}!={r['id'] for r in rows}:
        raise RuntimeError('Incomplete official question coverage')
    result=summarize(previous)
    result.update(protocol=protocol,arm=wrapped.arm,coverage_verified=True,
                  invalid_answers=sum(r['predicted'] is None for r in previous),
                  truncated=sum(r['truncated'] for r in previous),
                  mean_generated_tokens=sum(r['generated_tokens'] for r in previous)/len(previous),
                  gpu=torch.cuda.get_device_name(),engineering_smoke=args.smoke,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    if args.reason_then_choice:
        result.update(natural_generation_accuracy=sum(r['generated_correct'] for r in previous)/len(previous),
                      natural_invalid_answers=sum(r['generated_answer'] is None for r in previous))
    write_json(output/'metrics.json',result)
    print('FINAL',json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=RUN/'full/O2_seed42/epoch_4.pt')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--max-new-tokens',type=int,default=4096)
    parser.add_argument('--temperature',type=float,default=1.0)
    parser.add_argument('--presence-penalty',type=float,default=1.5)
    parser.add_argument('--no-thinking',action='store_true')
    parser.add_argument('--reason-then-choice',action='store_true')
    parser.add_argument('--only-aligned',action='store_true',
                        help='Evaluate one completed O2 checkpoint across two question shards; no native model is loaded')
    parser.add_argument('--single-arm',choices=['native','O2'],
                        help='Resume one comparison arm on one GPU without changing inference')
    args=parser.parse_args()
    rank=int(os.environ.get('LOCAL_RANK','0'))
    if rank not in (0,1):raise RuntimeError('This comparison uses at most two GPUs')
    if args.single_arm and (args.only_aligned or int(os.environ.get('WORLD_SIZE','1'))!=1):
        raise RuntimeError('Single-arm resume requires one GPU and no final sharding')
    if args.only_aligned and int(os.environ.get('WORLD_SIZE','1'))!=2:
        raise RuntimeError('Final aligned-only evaluation requires exactly two GPU shards')
    torch.cuda.set_device(rank);torch.set_num_threads(16)
    arm=('O2' if args.single_arm=='O2' else 'B1') if args.single_arm else (
        'O2' if args.only_aligned else 'B1' if rank==0 else 'O2')
    args.output.mkdir(parents=True,exist_ok=True)
    out=(args.output/'O2'/f'shard{rank}' if args.only_aligned
         else args.output/('native' if arm=='B1' else 'O2'))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():print('Already complete',out);return
    split='full_train' if args.smoke else 'test'
    rows=read_rows(RUN/'splits'/f'{split}.jsonl')
    rows=[r for r in rows if r['source']==('omni' if args.smoke else 'omni_test')]
    if args.smoke:
        selected=[]
        for task in sorted({r['task_type'] for r in rows}):
            selected.extend([r for r in rows if r['task_type']==task][:2])
        rows=selected
    else:
        assert len(rows)==1533 and len({r['id'] for r in rows})==1533
        complete=RUN/'full/O2_seed42/complete.json'
        if not complete.exists():raise RuntimeError('Final O2 checkpoint is not complete')
        if args.only_aligned:rows=rows[rank::2]
    tokens=torch.load(CACHE/('tokens.pt' if args.smoke else 'test_tokens.pt'),
                      map_location='cpu',weights_only=True,mmap=True)
    originals=[tokens[r['id']] for r in rows];validate_records(originals,rows)
    del tokens
    p=processor()
    records=[generation_record(r,t,p,not args.no_thinking) for r,t in zip(rows,originals)]
    # A meaningful contract: changing the answer does not change model inputs.
    changed=generation_record(dict(rows[0],answer='Z',target='Answer: Z'),originals[0],p,not args.no_thinking)
    assert torch.equal(changed['input_ids'],records[0]['input_ids'])
    if arm=='B1':
        model,_=load_qwen();wrapped=AlignedModel(model,'B1')
        choice_out=args.output/'native_choice';choice_out.mkdir(exist_ok=True)
        if not (choice_out/'metrics.json').exists():
            letters=json.loads((CACHE/('tokens_full_train_0_complete.json' if args.smoke else 'tokens_test_0_complete.json')).read_text())['answer_token_ids']
            predictions=score_native_durable(model,wrapped,originals,rows,letters,choice_out)
            result=summarize(predictions);result.update(arm='B1',baseline='pretrained',engineering_smoke=args.smoke)
            write_json(choice_out/'metrics.json',result)
    else:
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
        model,wrapped=load_model('O2',42);stage(model,wrapped,saved['stage'])
        restore_trainable(model,saved['adapter']);del saved
    model.eval()
    generate(model,wrapped,records,rows,p,args,out)


if __name__=='__main__':main()
