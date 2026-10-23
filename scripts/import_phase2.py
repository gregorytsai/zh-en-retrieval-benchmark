"""Import an explicitly supplied local phase-2 archive into an allowlisted public release.

Usage: python scripts/import_phase2.py /path/to/staging
Staging has batch1/ and batch2/ with official results. No model inference.
"""
from pathlib import Path
import json,sys,hashlib,statistics,csv,io,zipfile
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(source):
 out=ROOT/'data/phase2';out.mkdir(exist_ok=True)
 old=read(ROOT/'data/analysis.json'); models=read(ROOT/'data/models.json')
 result={'updated':'2026-10-24','rows':[],'datasets':{},'batches':{},'names':{k:v['short_name'] for k,v in models.items()},'aggregation':'Equal-weight dataset macro average; common configurations only. No cross-dataset pooled leaderboard.'}
 for batch,filename,expected in [('batch1','datasets.json',13278),('batch2','datasets-batch2.json',1200)]:
  src=source/batch;official=src/'official';manifest=read(official/'run-manifest.json');data=read(src/filename)
  assert digest(src/filename)==manifest['dataset_sha256']
  rows=[json.loads(l) for l in (official/'query-results.jsonl').read_text('utf-8').splitlines()]
  assert len(rows)==expected and all(r['status']=='ok' for r in rows)
  for k,v in manifest['model_sha256'].items():assert old['models'][k]['sha256']==v
  result['batches'][batch]={'rows':len(rows),'dataset_sha256':digest(src/filename),'source_files_sha256':{p.name:digest(p) for p in official.iterdir() if p.name in ['summary.json','query-results.jsonl','qrels.json','query-selection.json','paired-confidence.json','run-manifest.json','committed.json']}}
  for k,d in data['datasets'].items():
   result['datasets'][k]={'queries':len(d['queries']),'corpus':len(d['corpus']),'grade_mode':d['grade_mode'],'metadata':d['metadata'],'batch':batch}
  summaries=read(official/'summary.json')
  for s in summaries:
   rs=[r for r in rows if all(r[k]==s[k] for k in ['dataset','embedding','reranker'])]
   assert len(rs)==s['queries']
   s['total_ms_mean']=statistics.mean(r['total_ms'] for r in rs)
   s['weight_gib_sum']=old['models'][s['embedding']]['weight_gib']+(old['models'][s['reranker']]['weight_gib'] if s['reranker']!='standalone' else 0)
   result['rows'].append(s)
  # Only public numeric results, ranking IDs, qrels, selection and fixed metadata.
  with zipfile.ZipFile(ROOT/f'downloads/phase2-{batch}-results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
   for name in ['query-results.jsonl','qrels.json','query-selection.json','summary.json','paired-confidence.json','run-manifest.json','committed.json']:
    z.writestr(name,(official/name).read_bytes())
   z.writestr('NOTICE.txt','Research measurements and ranking IDs; dataset annotations retain their source licenses. See https://gregorytsai.github.io/zh-en-retrieval-benchmark/latest-methods.html#license . Source hashes are in data/phase2/analysis.json. No pilot or failed attempts included.\n')
  save(out/f'{batch}-confidence.json',read(official/'paired-confidence.json'))
  save(out/f'{batch}-manifest.json',manifest)
  # Keep source files in the original relative layout, without logs or machine paths.
  target=ROOT/'research/phase2'/batch
  target.mkdir(parents=True,exist_ok=True)
  code=src if batch=='batch1' else src/'runner'
  for p in code.glob('*.py'):
   if p.name=='legacy_run_benchmark.py':continue
   text=p.read_text('utf-8')
   assert '/Users/' not in text and '140.136.' not in text
   dest=target/('embedding-study-phase2-20261023/'+('batch2-runner/' if batch=='batch2' else '')+p.name)
   dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text,encoding='utf-8')
  legacy=target/'bilingual-retrieval-benchmark/run_benchmark.py';legacy.parent.mkdir(exist_ok=True)
  legacy.write_bytes((source/'batch1/legacy_run_benchmark.py').read_bytes())
 # Copy the frozen preparation sources used for MIRACL and its data manifest.
 for name in ['prepare_batch2_data.py','prepare_phase2_data.py']:
  (ROOT/'research/phase2/batch2/embedding-study-phase2-20261023'/name).write_bytes((source/'batch2'/name).read_bytes())
 save(out/'analysis.json',result)
 # Complete MLQA + ACORD adapted inputs; MIRACL full passages stay at upstream with a reproducible fetch helper.
 with zipfile.ZipFile(ROOT/'downloads/phase2-materials.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  z.writestr('batch1/datasets.json',(source/'batch1/datasets.json').read_bytes())
  b2=read(source/'batch2/datasets-batch2.json')
  for d in b2['datasets'].values():d['corpus']=[{'id':x['id'],'metadata':x.get('metadata',{})} for x in d['corpus']]
  z.writestr('batch2/selection-without-passage-text.json',json.dumps(b2,ensure_ascii=False))
  z.writestr('NOTICE.txt','MLQA: Facebook Research / mteb retrieval adapter, CC BY-SA 3.0, https://github.com/facebookresearch/MLQA and https://huggingface.co/datasets/mteb/MLQARetrieval . ACORD: The Atticus Project and ACORD authors, CC BY 4.0, https://huggingface.co/datasets/theatticusproject/acord . Adaptations: fixed selection, retrieval IDs, cluster metadata; ACORD stars mapped to grades 0-4. MIRACL topics/qrels: MIRACL authors, Apache 2.0, https://github.com/project-miracl/miracl . MIRACL Wikipedia passage text is omitted; retrieve from pinned upstream using research/phase2/fetch_miracl_materials.py. The benchmark code MIT license does not replace dataset licenses. License texts: https://creativecommons.org/licenses/by-sa/3.0/legalcode ; https://creativecommons.org/licenses/by/4.0/legalcode ; https://www.apache.org/licenses/LICENSE-2.0 .\n')
 fields=['dataset','embedding','reranker','queries','top1','ndcg10','mrr10','recall10','recall50','judged_coverage10','judged_coverage50','total_ms_mean','total_ms_p50','total_ms_p95','rerank_ms_p50','rerank_ms_p95','weight_gib_sum']
 with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(result['rows'])
 print('Imported',len(result['rows']),'groups;',sum(x['rows'] for x in result['batches'].values()),'measurements')
if __name__=='__main__':main(Path(sys.argv[1]))
