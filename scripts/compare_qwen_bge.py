"""Paired comparison of two embedders with the same Qwen 0.6B reranker.
No inference. Requires NumPy for passage-cluster bootstrap intervals.
"""
from pathlib import Path
import json,zipfile,statistics,collections,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
ORDER=['eng-zho','zho-zho','eng-eng','miracl-zh-zh','miracl-en-en']
def load(batch):
 with zipfile.ZipFile(R/f'downloads/matrix-{batch}-results.zip') as z:
  raw=z.read('query-results.jsonl')
  return {(r['dataset'],r['query_id']):r for r in map(json.loads,raw.splitlines()) if r['dataset'] in ORDER and r['reranker']=='qwen-reranker-0.6b'},hashlib.sha256(raw).hexdigest()
def main():
 q,qhash=load('B01');b,bhash=load('B05');assert q.keys()==b.keys() and len(q)==2556
 rng=np.random.default_rng(20261027);summary=[];pairs=[]
 for d in ORDER:
  keys=sorted(k for k in q if k[0]==d);diffs=collections.defaultdict(list);clusters=collections.defaultdict(list)
  for k in keys:
   x,y=q[k],b[k];assert x['cluster_id']==y['cluster_id']
   row={'dataset':d,'query_id':k[1],'cluster_id':x['cluster_id']}
   for m in ['ndcg10','top1','recall50','hit10']:
    row['qwen_'+m]=x[m];row['bge_'+m]=y[m];diffs[m].append(x[m]-y[m])
   clusters[x['cluster_id'] or k[1]].append(len(diffs['ndcg10'])-1);pairs.append(row)
  result={'dataset':d,'queries':len(keys),'clusters':len(clusters)}
  sizes=np.array([len(ix) for ix in clusters.values()]);draws=rng.integers(0,len(sizes),size=(10000,len(sizes)));den=sizes[draws].sum(axis=1)
  for m in diffs:
   v=np.array(diffs[m]);sums=np.array([v[ix].sum() for ix in clusters.values()]);boot=sums[draws].sum(axis=1)/den
   result[m]={'qwen':statistics.mean(q[k][m] for k in keys),'bge':statistics.mean(b[k][m] for k in keys),
     'delta':float(v.mean()),'qwen_better_queries':int((v>1e-12).sum()),'bge_better_queries':int((v< -1e-12).sum()),'ties':int((abs(v)<=1e-12).sum()),
     'paired_cluster_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
  summary.append(result)
 a=json.loads((R/'data/matrix/analysis.json').read_text('utf-8'))
 weights={e:a['models'][e]['weight_gib']+a['models']['qwen-reranker-0.6b']['weight_gib'] for e in ['qwen-4b','bge-m3']}
 out={'comparison':'Qwen 4B vs BGE-M3, both with Qwen 0.6B reranker','conditions':summary,'weights_gib':weights,
  'macro':{m:{'qwen':statistics.mean(r[m]['qwen'] for r in summary),'bge':statistics.mean(r[m]['bge'] for r in summary)} for m in diffs},
  'method':'Same 2,556 dataset-qualified queries; 5 task means equally weighted. Per-task paired passage-cluster percentile bootstrap: 10,000 resamples, seed 20261027. MIRACL clusters use stored query IDs. Intervals are descriptive, unadjusted for multiple comparisons. No overall bootstrap because cross-language dependence is not modeled. ACORD reranking excluded pending correction.',
  'source_rows_sha256':{'B01':qhash,'B05':bhash},'new_inference':False}
 (R/'data/matrix/qwen-bge-comparison.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
 (R/'data/matrix/qwen-bge-paired-results.jsonl').write_text('\n'.join(json.dumps(p) for p in pairs)+'\n',encoding='utf-8')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
