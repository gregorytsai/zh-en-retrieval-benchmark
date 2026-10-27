"""Import accepted 28-configuration outputs from an explicitly supplied archive.

Usage: python scripts/import_matrix.py PATH_TO_PROJECT
No inference. Raw rows/summaries are preserved; private runtime paths are removed
only from separately identified public metadata. Logs/attempts are never exported.
"""
from pathlib import Path
from collections import defaultdict
import json,hashlib,statistics,zipfile,csv,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text('utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('executable','run','source_root','output_root')}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,str) and x.startswith('/Users/'):return '[local path omitted]'
 return x
NOTICE='Accepted retrieval measurements; no pilots or invalidated attempts. Rows and summaries retain original bytes. Public manifests omit local runtime paths; original hashes remain in provenance.json. MLQA: Facebook Research, CC BY-SA 3.0 https://github.com/facebookresearch/MLQA ; ACORD: The Atticus Project and authors, CC BY 4.0 https://huggingface.co/datasets/theatticusproject/acord ; MIRACL topics/qrels: MIRACL authors, Apache 2.0 https://github.com/project-miracl/miracl . No MIRACL passage text or model weights included. See latest-methods.html#license and research/matrix/README.md.\n'
def pack(name,entries):
 entries=dict(entries);entries['NOTICE.txt']=NOTICE.encode();entries['LICENSES/Apache-2.0.txt']=(ROOT/'research/phase2/LICENSES/Apache-2.0.txt').read_bytes()
 with zipfile.ZipFile(ROOT/'downloads'/name,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,b in entries.items():
   info=zipfile.ZipInfo(n,(2026,10,27,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
def main(p):
 out=ROOT/'data/matrix';out.mkdir(exist_ok=True)
 old=read(ROOT/'data/analysis.json');previous=read(ROOT/'data/phase2/analysis.json')
 homes={'B01':p/'run-full-matrix/B01/official','B02':p/'run-full-matrix-continuation-v4/B02/combined',**{f'B{i:02d}':p/f'run-full-matrix-continuation-v4/B{i:02d}/segment-01/official' for i in range(3,8)}}
 a={'updated':'2026-10-27','completed':'2026-10-26T05:31:44.444+08:00','rows':[],'datasets':previous['datasets'],'names':previous['names'],'batches':{},'models':old['models'],'aggregation':'Six task means, equal weight 1/6; 28 complete configurations. Descriptive point estimates, no global confidence intervals.'}
 allrows=[];index=[];provenance={}
 for batch,h in homes.items():
  rows=[json.loads(l) for l in (h/'query-results.jsonl').read_text('utf-8').splitlines()];summaries=read(h/'summary.json')
  assert len(rows)==10452 and len(summaries)==24 and all(r['status']=='ok' for r in rows)
  allrows.extend(rows);groups=defaultdict(list)
  for r in rows:groups[(r['dataset'],r['embedding'],r['reranker'])].append(r)
  for s in summaries:
   g=groups[(s['dataset'],s['embedding'],s['reranker'])]
   s=dict(s,total_ms_mean=statistics.mean(r['total_ms'] for r in g),weight_gib_sum=old['models'][s['embedding']]['weight_gib']+(old['models'][s['reranker']]['weight_gib'] if s['reranker']!='standalone' else 0))
   a['rows'].append(s)
  raw={n:(h/n).read_bytes() for n in ['query-results.jsonl','summary.json']}
  a['batches'][batch]={'rows':len(rows),'source_files_sha256':{n:digest(h/n) for n in raw}}
  for n in ['run-manifest.json','batch-completion-audit.json','committed.json','index-stats.jsonl']:
   if not (h/n).exists():continue
   provenance[f'{batch}/{n}']={'original_sha256':digest(h/n),'public_metadata':'local paths removed where present'}
   if n.endswith('.jsonl'):
    ix=[json.loads(l) for l in (h/n).read_text('utf-8').splitlines()];index.extend(ix)
    raw[n]=('\n'.join(json.dumps(clean(x),ensure_ascii=False) for x in ix)+'\n').encode()
   else:raw[n]=(json.dumps(clean(read(h/n)),ensure_ascii=False,indent=2)+'\n').encode()
  if batch=='B02':
   for child in ['B02-A','B02-B']:
    ch=p/f'run-full-matrix-continuation-v3/{child}/segment-01/official'
    for n in ['run-manifest.json','subbatch-completion-audit.json','index-stats.jsonl']:
     if not (ch/n).exists():continue
     provenance[f'{child}/{n}']={'original_sha256':digest(ch/n),'public_metadata':'local paths removed where present'}
     if n.endswith('.jsonl'):
      ix=[json.loads(l) for l in (ch/n).read_text('utf-8').splitlines()];index.extend(ix);raw[f'{child}/{n}']=('\n'.join(json.dumps(clean(x),ensure_ascii=False) for x in ix)+'\n').encode()
     else:raw[f'{child}/{n}']=(json.dumps(clean(read(ch/n)),ensure_ascii=False,indent=2)+'\n').encode()
  pack(f'matrix-{batch}-results.zip',raw)
 shared=homes['B07']
 pack('matrix-shared.zip',{n:(shared/n).read_bytes() for n in ['qrels.json','query-selection.json']})
 a['shared_sha256']={n:digest(shared/n) for n in ['qrels.json','query-selection.json']}
 def pct(v,q):
  v=sorted(v);i=(len(v)-1)*q;lo=int(i);return v[lo]+(v[min(lo+1,len(v)-1)]-v[lo])*(i-lo)
 a['reranker_timing']={}
 for rr in ['bge-reranker','qwen-reranker-0.6b','qwen-reranker-4b']:
  vals=[r['rerank_ms'] for r in allrows if r['reranker']==rr]
  a['reranker_timing'][rr]={'rows':len(vals),'p50_ms':pct(vals,.5),'p95_ms':pct(vals,.95),'mean_macro_ndcg10':statistics.mean(s['ndcg10'] for s in a['rows'] if s['reranker']==rr)}
 a['embedder_timing']={}
 for e in {r['embedding'] for r in allrows}:
  vals=[r['embedding_ms'] for r in allrows if r['embedding']==e and r['reranker']=='standalone']
  a['embedder_timing'][e]={'rows':len(vals),'p50_ms':pct(vals,.5),'p95_ms':pct(vals,.95)}
 save(out/'analysis.json',a);save(out/'provenance.json',provenance);save(out/'index-stats.json',clean(index))
 save(out/'original-ranking.json',read(p/'run-full-matrix-continuation-v4/global-ranking.json'))
 fields=['dataset','embedding','reranker','queries','top1','ndcg10','mrr10','recall10','recall50','candidate_recall50','judged_coverage50','total_ms_mean','total_ms_p50','total_ms_p95','rerank_ms_p50','rerank_ms_p95','weight_gib_sum']
 with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(a['rows'])
 # Complete original runner sources, preserving relative layout; no private logs.
 dest=ROOT/'research/matrix/workspace/embedding-study-phase2-20261023'
 for sub in ['full-matrix-runner','full-matrix-runner-continuation-v3','full-matrix-runner-continuation-v4']:
  (dest/sub).mkdir(parents=True,exist_ok=True)
  for file in (p/sub).glob('*.py'):
   assert '/Users/' not in file.read_text('utf-8')
   shutil.copyfile(file,dest/sub/file.name)
 for n in ['prepare_phase2_data.py','prepare_batch2_data.py','fetch_miracl_missing.py']:
  shutil.copyfile(p/n,dest/n)
 shutil.copyfile(ROOT/'research/phase2/fetch_miracl_materials.py',ROOT/'research/matrix/fetch_miracl_materials.py')
 print('Imported 28 configurations,',len(a['rows']),'groups,',len(allrows),'rows; 8 result archives')
if __name__=='__main__':main(Path(sys.argv[1]))
