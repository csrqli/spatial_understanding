"""Prepare existing MindCube rationale annotations without adding test content.

The 3DThinker export's integer idx is not the raw MindCube row index. Only
the audited exact question/image/answer matches may provide a rationale.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from .config import CACHE, ROOT, RUN
from .generation_compare import generation_record, SYSTEM
from .runtime import processor, read_rows

V3=ROOT/'results/qwen35_omega_omnispatial/20261026_improve_v3'


def main():
    destination=V3/'data'
    output=destination/'reasoning_tokens.pt'
    if output.exists():raise FileExistsError('Preserve prepared rationale token provenance')
    rationales=read_rows(destination/'verified_mind_reasoning.jsonl')
    annotations={r['mind_id']:r for r in rationales}
    assert len(annotations)==len(rationales)==10000
    rows={r['id']:r for r in read_rows(RUN/'splits/full_train.jsonl') if r['source']=='mind'}
    split_of={}
    for split in ('train','development','confirmation'):
        for r in read_rows(RUN/'splits'/f'{split}.jsonl'):
            if r['source']=='mind':split_of[r['id']]=split
    assert set(rows)==set(split_of)==set(annotations)
    old_tokens=torch.load(CACHE/'tokens.pt',map_location='cpu',weights_only=True,mmap=True)
    p=processor();prepared={};metadata=[];rejected=Counter();counts=Counter()
    start=time.monotonic()
    for number,(rid,row) in enumerate(rows.items()):
        annotation=annotations[rid];assert annotation['answer']==row['answer']
        old=old_tokens[rid]
        prefix=generation_record(row,old,p,thinking=True)
        reasoning=annotation['reasoning']
        # Prefix is the same system/user prompt as the generation diagnostic.
        # Verify tokenization across the concatenation boundary explicitly.
        question=row['prompt']
        content=[]
        for i in range(len(row['image_hashes'])):
            if len(row['image_hashes'])>1:content.append({'type':'text','text':f'Image {i+1}: '})
            content.append({'type':'image'})
        from .config import FORMAT
        assert question.endswith(FORMAT)
        content.append({'type':'text','text':question[:-len(FORMAT)].rstrip()+'\n\nSelect one valid option. End with Answer: X.'})
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':content},
                  {'role':'assistant','reasoning_content':reasoning,'content':'Answer: '+row['answer']}]
        text=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=True)
        replacements=[p.replace_image_token({'image_grid_thw':old['grid'].long()},i)
                      for i in range(len(row['image_hashes']))]
        expanded,_=p.get_text_with_replacements([text],replacements,[],[])
        ids=torch.tensor(p.tokenizer(expanded[0],add_special_tokens=False).input_ids,dtype=torch.int32)
        n=prefix['prompt_length']
        assert torch.equal(ids[:n],prefix['input_ids']),rid
        answer_position=len(ids)-3
        answer_token=p.tokenizer.encode('Answer: '+row['answer'],add_special_tokens=False)[-1]
        assert int(ids[answer_position])==answer_token and answer_position>=n
        if len(ids)>8192:rejected['over_8192_tokens']+=1;continue
        new_id=rid+'/reasoning'
        record=dict(old,id=new_id,original_id=rid,input_ids=ids,prompt_length=n,
                    suffix_length=len(ids)-n,answer_position=answer_position,
                    annotation_variant='reasoning',training_partition=split_of[rid])
        prepared[new_id]=record
        metadata.append({'id':new_id,'original_id':rid,'source':'mind','group_id':row['group_id'],
                         'image_hashes':row['image_hashes'],'feature_key':row['feature_key'],
                         'answer':row['answer'],'partition':split_of[rid],
                         'prompt_length':n,'supervised_tokens':len(ids)-n,'total_tokens':len(ids)})
        counts[split_of[rid]]+=1
        if (number+1)%1000==0:print('prepared_rationales',number+1,'/',len(rows),flush=True)
    # No new images or additional unique questions: these are annotation variants.
    temporary=output.with_suffix('.pt.tmp');torch.save(prepared,temporary);temporary.replace(output)
    (destination/'reasoning_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata))
    report={'raw_annotations':len(annotations),'prepared':len(prepared),'partitions':dict(counts),
            'rejected':dict(rejected),'max_tokens':max(r['total_tokens'] for r in metadata),
            'mean_supervised_tokens':sum(r['supervised_tokens'] for r in metadata)/len(metadata),
            'input_construction_independent_of_answers':True,
            'test_questions_included':0,'additional_unique_questions':0,
            'source_manifest_sha256':hashlib.sha256((destination/'verified_mind_reasoning.jsonl').read_bytes()).hexdigest(),
            'elapsed_seconds':time.monotonic()-start,
            'note':'Preserves existing image-disjoint train/development/confirmation assignment; a model already refit on all V2 partitions has seen these images.'}
    (destination/'reasoning_tokens_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
