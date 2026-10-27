"""Prepare all designated training data; preserve image order and split isolation."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
import hashlib
import io
import json
from pathlib import Path
import random
import re
import threading
import uuid
import pyarrow.parquet as pq
from PIL import Image
from research.qwen35_omega.data import canonical_rgb
from .config import DATA, FORMAT, MIND, RUN, SAT, SAT_REVISION, SEED, V1


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


class Images:
    def __init__(self):
        self.cache, self.metadata, self.paths = {}, {}, {}
        self.lock = threading.Lock()
        self.folder = DATA/'images'
        self.folder.mkdir(parents=True, exist_ok=True)

    def put(self, raw):
        key = hashlib.sha256(raw).hexdigest()
        with self.lock:
            owner = key not in self.cache
            if owner:self.cache[key]=Future()
            future=self.cache[key]
        if not owner:return future.result()
        try:
            with Image.open(io.BytesIO(raw)) as im:
                rgb = canonical_rgb(im)
                rgb.load()
                w,h = rgb.size
                digest = hashlib.sha256(f'RGB:{w}:{h}:'.encode()+rgb.tobytes()).hexdigest()
                path = self.folder/f'{digest}.png'
                if not path.exists():path=self.folder/f'{digest}.img'
                if not path.exists():
                    tmp = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
                    # Preserve encoded source bytes; all consumers apply the
                    # same canonical RGB conversion used for the pixel digest.
                    tmp.write_bytes(raw)
                    tmp.replace(path)
            with self.lock:
                self.metadata[digest] = {'image_sha256':digest, 'image':str(path), 'width':w, 'height':h}
            future.set_result(digest)
            return digest
        except BaseException as error:
            future.set_exception(error)
            raise

    def file(self, path):
        key=str(path)
        with self.lock:
            if key in self.paths:return self.paths[key]
        digest=self.put(path.read_bytes())
        with self.lock:self.paths[key]=digest
        return digest


class Union:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x,x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def join(self, a, b):
        a,b = self.find(a), self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)


def main():
    destination = RUN/'splits'
    if (destination/'manifest.json').exists():
        raise FileExistsError('Frozen V2 split manifest already exists')
    store, groups = Images(), Union()
    pools = defaultdict(list)
    source_counts, rejected = Counter(), Counter()
    source_hashes = {}
    forbidden = set()
    omni_source=Path(json.loads((V1/'splits/content_groups/manifest.json').read_text())['source'])
    omni_options={r['task_type']+'/'+r['id']:[chr(65+i) for i,v in enumerate(r['options']) if v]
                  for r in json.loads(omni_source.read_text())}
    # These four released questions encode the option labels in the picture.
    # All four source images were visually inspected during V2 preparation.
    visual_options={'Spatial_Interaction/2064_0':['A','B'],'Spatial_Interaction/2070_0':['A','B'],
                    'Spatial_Interaction/2071_0':['A','B'],'Spatial_Interaction/2084_0':['A','B','C']}
    for key,values in visual_options.items():
        assert not omni_options[key]
        omni_options[key]=values
    def bind(row):
        first = row['image_hashes'][0]
        for digest in row['image_hashes']: groups.join(first,digest)
        if row.get('scene_id'): groups.join(first, 'scene:'+row['source']+':'+row['scene_id'])
        row['images'] = [store.metadata[h]['image'] for h in row['image_hashes']]
        row['target'] = 'Answer: '+row['answer']
        row['feature_key'] = hashlib.sha256(json.dumps(row['image_hashes'],separators=(',',':')).encode()).hexdigest()
        pools[row.pop('pool')].append(row)

    # Existing Omni train/development assignments are retained, so the common
    # B1 pilot warm-start never saw the retained development labels.
    for split in ('train','development','confirmation'):
        path = V1/'splits/content_groups'/f'{split}.jsonl'
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        records=read(path)
        with ThreadPoolExecutor(max_workers=12) as executor:
            digests=list(executor.map(store.file,[Path(r['image']) for r in records]))
        for old,digest in zip(records,digests):
            assert digest == old['image_sha256']
            assert old['answer'] in omni_options[old['id']]
            bind({'id':'omni/'+old['id'],'source':'omni','task_type':old['task_type'],
                  'sub_task_type':old['sub_task_type'],'image_hashes':[digest],
                  'prompt':old['prompt']+'\n\n'+FORMAT,'answer':old['answer'],
                  'choices':omni_options[old['id']],'pool':'omni_'+split})
        print('prepared omni',split,len(pools['omni_'+split]),flush=True)
    # Test pixels only, not test questions/labels, are used for overlap exclusion.
    test = Path('/dataset1/yu025/datasets/omnispatial/OmniSpatial/extracted/test/OmniSpatial-test')
    with ThreadPoolExecutor(max_workers=12) as executor:
        forbidden.update(executor.map(store.file, sorted(test.glob('*/*.png'))))

    for split in ('train','val','test'):
        for path in sorted((SAT/'data').glob(split+'-*.parquet')):
            source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            columns = ['images'] if split == 'test' else None
            number = 0
            with ThreadPoolExecutor(max_workers=12) as executor:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
                    raw_rows = batch.to_pylist()
                    flat = [i['bytes'] for r in raw_rows for i in r['images']]
                    if any(v is None for v in flat): raise ValueError('SAT image bytes missing')
                    hashes = iter(executor.map(store.put,flat))
                    for old in raw_rows:
                        images = [next(hashes) for _ in old['images']]
                        if not images: raise ValueError('Empty image list')
                        source_counts['sat_'+split] += 1
                        number += 1
                        if split == 'test':
                            forbidden.update(images)
                            continue
                        answers = old['answers']
                        if not 2 <= len(answers) <= 4 or len(set(answers)) != len(answers):
                            rejected['sat_invalid_choices'] += 1
                            continue
                        if answers.count(old['correct_answer']) != 1:
                            raise ValueError('SAT correct answer is not unique')
                        rid = f'sat/{split}/{path.stem}/{number}'
                        # SAT choices can place the correct response first.
                        # Deterministic per-record shuffling prevents that shortcut.
                        order = list(range(len(answers)))
                        rng = random.Random(int(hashlib.sha256(rid.encode()).hexdigest()[:16],16))
                        rng.shuffle(order)
                        options = [answers[i] for i in order]
                        answer = chr(65+options.index(old['correct_answer']))
                        prompt = old['question']+'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(options))+'\n\n'+FORMAT
                        bind({'id':rid,'source':'sat','task_type':old['question_type'] or 'unspecified',
                              'sub_task_type':old['question_type'] or 'unspecified','image_hashes':images,
                              'prompt':prompt,'answer':answer,'choices':[chr(65+i) for i in range(len(options))],
                              'pool':'sat_'+split,'original_question':old['question'],
                              'original_options':answers,'answer_text':old['correct_answer'],'option_permutation':order})
            print('prepared SAT',path.name,number,'unique images',len(store.metadata),flush=True)

    # MindCube's train and tiny-benchmark scenes are distinct in the local
    # release. 3DThinker-10K is the same question set, not 10K additional QAs.
    tiny = MIND/'raw/MindCube_tinybench.jsonl'
    source_hashes[str(tiny)] = hashlib.sha256(tiny.read_bytes()).hexdigest()
    tiny_scenes = set()
    for old in read(tiny):
        for image in old['images']:
            tiny_scenes.add(str(Path(image).parent))
            forbidden.add(store.file(MIND/image))
    path = MIND/'raw/MindCube_train.jsonl'
    source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    records=read(path)
    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(store.file,sorted({MIND/i for r in records for i in r['images']})))
    for old in records:
        scenes = {str(Path(i).parent) for i in old['images']}
        if scenes & tiny_scenes:
            rejected['mind_tiny_scene_overlap'] += 1
            continue
        if len(scenes) != 1: raise ValueError('MindCube record crosses source scenes')
        if old['gt_answer'] not in 'ABCD': raise ValueError('Unknown MindCube answer')
        bind({'id':'mind/'+old['id'],'source':'mind','task_type':str(old['category'][1]),
              'sub_task_type':'/'.join(map(str,old['category'])),'scene_id':next(iter(scenes)),
              'image_hashes':[store.file(MIND/i) for i in old['images']],
              'prompt':old['question']+'\n\n'+FORMAT,'answer':old['gt_answer'],
              'choices':['A','B','C','D'],'pool':'mind_train'})
    print('prepared MindCube',len(pools['mind_train']),flush=True)

    # Connected components prevent a frame shared by two multi-image records
    # from crossing partitions. Official test content takes precedence.
    forbidden_components = {groups.find(x) for x in forbidden}
    heldout_components = {groups.find(r['image_hashes'][0]) for p in ('omni_development','omni_confirmation','sat_val') for r in pools[p]}
    clean = defaultdict(list)
    for pool, rows in pools.items():
        for row in rows:
            component = groups.find(row['image_hashes'][0])
            row['group_id'] = hashlib.sha256(component.encode()).hexdigest()
            if component in forbidden_components:
                rejected[pool+'_test_overlap'] += 1
            elif pool.endswith('_train') and component in heldout_components:
                rejected[pool+'_heldout_overlap'] += 1
            else: clean[pool].append(row)
    # Split source-defined SAT validation image components and MindCube train
    # scene components deterministically. Newly reserved confirmation stays sealed.
    rng = random.Random(SEED)
    def assignments(rows, fractions):
        keys = sorted({r['group_id'] for r in rows});rng.shuffle(keys)
        a,b = int(len(keys)*fractions[0]),int(len(keys)*fractions[1])
        if sum(fractions) == 1: b = len(keys)-a
        return {k:('development' if i<a else 'confirmation' if i<a+b else 'train') for i,k in enumerate(keys)}
    partitions = defaultdict(list)
    sat_assignment = assignments(clean['sat_val'],(.5,.5))
    mind_assignment = assignments(clean['mind_train'],(.05,.05))
    for split in ('train','development','confirmation'):
        partitions[split].extend(clean['omni_'+split])
    partitions['train'].extend(clean['sat_train'])
    for r in clean['sat_val']: partitions[sat_assignment[r['group_id']]].append(r)
    for r in clean['mind_train']: partitions[mind_assignment[r['group_id']]].append(r)
    # Remove exact repeated question/image/answer records inside each split.
    for split, rows in list(partitions.items()):
        seen, result = set(), []
        for r in rows:
            key = (tuple(r['image_hashes']),re.sub(r'\s+',' ',r.get('original_question',r['prompt'])),r.get('answer_text',r['answer']),tuple(sorted(r.get('original_options',[]))))
            if key in seen: rejected[split+'_duplicate_qa'] += 1
            else: seen.add(key); result.append(r)
        partitions[split] = result
    for a,b in (('train','development'),('train','confirmation'),('development','confirmation')):
        assert not ({r['group_id'] for r in partitions[a]} & {r['group_id'] for r in partitions[b]}), (a,b)
    for split, rows in partitions.items(): write(destination/f'{split}.jsonl',rows)
    # Full refits include the declared training-side partitions, after design lock.
    full = sum([partitions[s] for s in ('train','development','confirmation')],[])
    write(destination/'full_train.jsonl',full)
    # A deterministic, source-stratified 12K pilot; no held-out examples.
    pilot = []
    for source, cap in (('sat',8000),('omni',2000),('mind',2000)):
        rows = [r for r in partitions['train'] if r['source']==source]
        rng.shuffle(rows);pilot.extend(rows[:cap])
    write(destination/'pilot_train.jsonl',pilot)
    # Tiny fitting probe uses 64 distinct training records, balanced by source.
    tiny_rows=[]
    for source,n in (('sat',24),('omni',20),('mind',20)):
        tiny_rows += [r for r in pilot if r['source']==source][:n]
    write(destination/'tiny_train.jsonl',tiny_rows)
    evaluation=[]
    for source,cap in (('sat',512),('omni',512),('mind',256)):
        rows=[r for r in partitions['development'] if r['source']==source]
        rng.shuffle(rows);evaluation.extend(rows[:cap])
    write(destination/'pilot_development.jsonl',evaluation)
    used_hashes={h for r in full for h in r['image_hashes']}
    write(destination/'images.jsonl',[store.metadata[h] for h in sorted(used_hashes)])
    sequences={r['feature_key']:{'feature_key':r['feature_key'],'images':r['images'],'image_hashes':r['image_hashes']} for r in full}
    write(destination/'sequences.jsonl',[sequences[k] for k in sorted(sequences)])
    manifest={'schema':'q35_expanded_spatial_v2','seed':SEED,'sat_revision':SAT_REVISION,
              'raw_sat_counts':dict(source_counts),'rejections':dict(rejected),'source_sha256':source_hashes,
              'counts':{s:len(v) for s,v in partitions.items()},'full_train':len(full),'pilot_train':len(pilot),
              'sources':{s:dict(Counter(r['source'] for r in rows)) for s,rows in partitions.items()},
              'task_counts':{s:dict(Counter(r['source']+'/'+r['task_type'] for r in rows)) for s,rows in partitions.items()},
              'answer_counts':{s:dict(Counter(r['answer'] for r in rows)) for s,rows in partitions.items()},
              'image_count':len(used_hashes),'sequence_count':len(sequences),
              'frames_per_record':dict(Counter(len(r['images']) for r in full)),
              'test_labels_read':False,'mind_tiny_labels_used':False,'visual_option_annotations':visual_options,
              'option_metadata_finalized':True,
              'grouping':'Connected decoded-image components; additionally source-scene directories for MindCube.',
              'limitations':['SAT parquet lacks physical-scene IDs; exact image-component separation is not proof of independent physical scenes.',
                             'OmniSpatial test was observed in V1; its V2 result is follow-up evidence, not a new blind test.',
                             'HSTAR action dialogues are excluded from this multiple-choice QA recipe.',
                             '3DThinker-10K duplicates MindCube-train and is not counted as additional data.'],
              'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.jsonl')}}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
