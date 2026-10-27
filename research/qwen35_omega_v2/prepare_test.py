"""Open the sealed SAT/OmniSpatial test labels after the design is locked."""
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
from .config import DATA,FORMAT,RUN,SAT,TRAINING_SEEDS
from .prepare import Images,write


def bind(row,store):
    row['images']=[store.metadata[h]['image'] for h in row['image_hashes']]
    row['target']='Answer: '+row['answer']
    row['feature_key']=hashlib.sha256(json.dumps(row['image_hashes'],separators=(',',':')).encode()).hexdigest()
    return row


def omni_choices(row):
    # The released Complex Logic items can put A-D choices in the image,
    # leaving the textual options list empty. Keep these benchmark questions.
    options=row['options']
    choices=[chr(65+i) for i in range(len(options))] if options else list('ABCD')
    if not 2<=len(choices)<=4 or not 0<=row['answer']<len(choices):
        raise ValueError(f"Invalid OmniSpatial choice metadata: {row['task_type']}/{row['id']}")
    return choices


def main():
    split=RUN/'splits'
    lock_path=RUN/'manifests/design_lock.json'
    if not lock_path.exists():raise RuntimeError('Design is not locked; refusing to open test labels')
    lock=json.loads(lock_path.read_text())
    if tuple(lock['seeds'])!=TRAINING_SEEDS:
        raise RuntimeError('Only the single fixed training seed is authorized')
    for arm in lock['full_arms']:
        for seed in lock['seeds']:
            if not (RUN/'full'/f'{arm}_seed{seed}'/'complete.json').exists():
                raise RuntimeError(f'Test labels remain sealed until every full refit completes: {arm} seed {seed}')
    if (split/'test.jsonl').exists():raise FileExistsError('Test artifacts already exist; preserve the locked test set')
    store=Images();rows=[];source_files={}
    # This test set was previously seen during the V1 follow-up, but its V2
    # rows are opened only after the V2 design and full refits are locked.
    root=Path('/dataset1/yu025/datasets/omnispatial/OmniSpatial/extracted/test/OmniSpatial-test')
    source=root/'data.json';source_files[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    official_omni=json.loads(source.read_text())
    for old in official_omni:
        category=old['task_type'];image=root/category/f"{old['id'].split('_')[0]}.png"
        digest=store.file(image)
        choices=omni_choices(old)
        prompt=old['question']+'\n'+'\n'.join(f'{c}. {v}' for c,v in zip(choices,old['options']))+'\n\n'+FORMAT
        rows.append(bind({'id':'omni_test/'+category+'/'+old['id'],'source':'omni_test','task_type':category,
            'sub_task_type':old['sub_task_type'],'group_id':'test/'+digest,'image_hashes':[digest],
            'prompt':prompt,'answer':chr(65+old['answer']),'choices':choices,
            'options_in_image':not bool(old['options'])},store))
    if len(rows)!=len(official_omni):raise RuntimeError('Incomplete OmniSpatial test coverage')
    # Keep every rotation for SAT, and aggregate rotations by question at
    # scoring time. This is a declared V2 circular-consistency protocol.
    sat_path=SAT/'data/test-00000-of-00001.parquet'
    source_files[str(sat_path)]=hashlib.sha256(sat_path.read_bytes()).hexdigest()
    base_count=0;rejected=Counter();rngseed=20261025
    for batch in pq.ParquetFile(sat_path).iter_batches(batch_size=256):
        for old in batch.to_pylist():
            answers=old['answers']
            if not 2<=len(answers)<=4 or len(set(answers))!=len(answers) or answers.count(old['correct_answer'])!=1:
                rejected['invalid_choices']+=1;continue
            hashes=[store.put(im['bytes']) for im in old['images']]
            rid=f"sat_test/{base_count:05d}";base_count+=1
            order=list(range(len(answers)));random.Random(hashlib.sha256(rid.encode()).hexdigest()).shuffle(order)
            ordered=[answers[i] for i in order]
            for shift in range(len(ordered)):
                option_text=ordered[shift:]+ordered[:shift]
                choices=[chr(65+i) for i in range(len(option_text))]
                answer=chr(65+option_text.index(old['correct_answer']))
                prompt=old['question']+'\n'+'\n'.join(f'{c}. {v}' for c,v in zip(choices,option_text))+'\n\n'+FORMAT
                rows.append(bind({'id':f'{rid}/rotation{shift}','circular_id':rid,'source':'sat_test',
                    'task_type':old['question_type'] or 'unspecified','sub_task_type':old['question_type'] or 'unspecified',
                    'group_id':'test/'+hashes[0],'image_hashes':hashes,'prompt':prompt,'answer':answer,
                    'choices':choices},store))
    if len({r['id'] for r in rows})!=len(rows):raise RuntimeError('Duplicate test question IDs')
    used={h for r in rows for h in r['image_hashes']}
    write(split/'test.jsonl',rows)
    write(split/'test_images.jsonl',[store.metadata[h] for h in sorted(used)])
    manifest={'schema':'qwen35_expanded_spatial_v2_sealed_test','design_lock_sha256':hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        'test_labels_opened_after_full_training':True,'sat_questions':base_count,
        'sat_rotation_rows':sum(r['source']=='sat_test' for r in rows),
        'omni_questions':sum(r['source']=='omni_test' for r in rows),'rows':len(rows),
        'omni_expected_questions':len(official_omni),
        'omni_image_option_questions':sum(r.get('options_in_image',False) for r in rows),
        'omni_task_counts':dict(Counter(r['task_type'] for r in rows if r['source']=='omni_test')),
        'unique_question_ids':len({r['id'] for r in rows}),'coverage_complete':True,
        'unique_images':len(used),'rejections':dict(rejected),'sources':source_files,
        'protocol':'All cyclic option rotations for SAT test; OmniSpatial official test order unchanged; report SAT per-question mean and strict all-rotations accuracy separately.'}
    (split/'test_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
