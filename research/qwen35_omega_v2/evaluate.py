"""Choice scoring after a fixed answer prefix, without supplying the true letter."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
from .config import CACHE,RUN,ARMS,TRAINING_SEEDS
from .cache import make_batch
from .runtime import load_model,stage,restore_trainable,read_rows,load_qwen,AlignedModel


def summarize(rows):
    tasks=defaultdict(list)
    for r in rows:tasks[(r['source'],r['task_type'])].append(r['correct'])
    sources=defaultdict(dict)
    for (source,task),values in sorted(tasks.items()):sources[source][task]={'n':len(values),'accuracy':sum(values)/len(values)}
    source_scores={s:{'macro':sum(v['accuracy'] for v in t.values())/len(t),
                      'micro':sum(v['accuracy']*v['n'] for v in t.values())/sum(v['n'] for v in t.values()),
                      'n':sum(v['n'] for v in t.values()),'tasks':t} for s,t in sources.items()}
    return {'n':len(rows),'source_macro_mean':sum(v['macro'] for v in source_scores.values())/len(source_scores),
            'micro_accuracy':sum(r['correct'] for r in rows)/len(rows),'sources':source_scores,
            'protocol':'Argmax over valid option letters following the fixed Answer: prefix; no ground-truth letter in model input.'}


def validate_records(records,rows):
    if len(records)!=len(rows) or len({row['id'] for row in rows})!=len(rows):
        raise RuntimeError('Evaluation requires unique IDs and exactly one token record per question')
    for record,row in zip(records,rows):
        for key in ('id','source','task_type','sub_task_type','image_hashes','feature_key','answer','choices'):
            if record[key]!=row[key]:
                raise RuntimeError(f"Cached question mismatch for {row['id']}: {key}")


@torch.no_grad()
def score(model,wrapped,records,rows,arm,letters,batch_size=8):
    predictions=[]
    model.eval()
    for begin in range(0,len(records),batch_size):
        part=records[begin:begin+batch_size]
        batch,context=make_batch(part,arm,mode='choice')
        assert 'labels' not in batch
        wrapped.context=context
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(**{k:v.cuda() for k,v in batch.items()},logits_to_keep=1,use_cache=False).logits[:,-1].float()
        finally:wrapped.context=None
        for i,(record,row) in enumerate(zip(part,rows[begin:begin+batch_size])):
            ids=torch.tensor([letters[a] for a in record['choices']],device='cuda')
            values=logits[i,ids]
            selected=record['choices'][int(values.argmax())]
            answer_index=record['choices'].index(record['answer'])
            result={k:row[k] for k in ('id','source','task_type','sub_task_type','group_id','answer')}
            result.update(predicted=selected,correct=selected==record['answer'],
                          choice_logits=values.cpu().tolist(),choice_letters=record['choices'],
                          choice_nll=float(-values.log_softmax(0)[answer_index]),
                          unconstrained_top1_is_choice=int(logits[i].argmax()) in ids.tolist())
            if 'circular_id' in row:result['circular_id']=row['circular_id']
            predictions.append(result)
        if begin%(batch_size*20)==0:print('scored',len(predictions),'/',len(records),flush=True)
    return predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS),required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--baseline',choices=('warm','pretrained'));p.add_argument('--split',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    if (args.checkpoint is None)==(args.baseline is None):raise ValueError('Specify exactly one trained checkpoint or fixed baseline')
    if args.baseline and args.arm!='B1':raise ValueError('Fixed baselines have no spatial adapter')
    if (args.output/'metrics.json').exists():print('Evaluation already complete');return
    if args.split=='test':
        lock=json.loads((RUN/'manifests/design_lock.json').read_text())
        if tuple(lock['seeds'])!=TRAINING_SEEDS:
            raise RuntimeError('Only the single fixed training seed is authorized')
        for arm in lock['full_arms']:
            for seed in lock['seeds']:
                if not (RUN/'full'/f'{arm}_seed{seed}'/'complete.json').exists():raise RuntimeError('Test locked until every declared full refit completes')
    rows=read_rows(RUN/'splits'/f'{args.split}.jsonl')
    name='test' if args.split=='test' else 'full_train'
    tokens=torch.load(CACHE/('test_tokens.pt' if name=='test' else 'tokens.pt'),map_location='cpu',weights_only=True)
    records=[tokens[r['id']] for r in rows];del tokens
    validate_records(records,rows)
    letters=json.loads((CACHE/f'tokens_{name}_0_complete.json').read_text())['answer_token_ids']
    if args.baseline=='pretrained':
        model,_=load_qwen();wrapped=AlignedModel(model,'B1')
    elif args.baseline=='warm':model,wrapped=load_model('B1',42)
    else:
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert saved['configuration']['arm']==args.arm
        model,wrapped=load_model(args.arm,saved['configuration']['seed'])
        stage(model,wrapped,saved['stage']);restore_trainable(model,saved['adapter']);del saved
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    predictions=score(model,wrapped,records,rows,args.arm,letters,args.batch_size)
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    report=summarize(predictions)
    report.update(arm=args.arm,baseline=args.baseline,checkpoint=str(args.checkpoint),split=args.split,elapsed_seconds=time.monotonic()-start,
                  gpu=torch.cuda.get_device_name(),batch_size=args.batch_size,cache_records_validated=True)
    if args.split=='test':
        circle=defaultdict(list)
        for r in predictions:
            if 'circular_id' in r:circle[r['circular_id']].append(r)
        if circle:
            for key,values in circle.items():assert len(values)==len(values[0]['choice_letters']),key
            report['sat_circular']={'questions':len(circle),'all_rotations_correct_accuracy':sum(all(r['correct'] for r in v) for v in circle.values())/len(circle),
                                    'rotation_mean_accuracy':sum(sum(r['correct'] for r in v)/len(v) for v in circle.values())/len(circle)}
            collapsed=[r for r in predictions if 'circular_id' not in r]
            for key,values in circle.items():collapsed.append(dict(values[0],id=key,correct=sum(r['correct'] for r in values)/len(values)))
            report['per_question']=summarize(collapsed)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
