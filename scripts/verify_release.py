"""Offline checks: hashes, all ranking metrics, summaries, datasets and HTML links."""
from pathlib import Path
import json,zipfile,hashlib,collections,math,statistics
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def require(test,message):
 if not test:raise AssertionError(message)
def metrics(ids,relevant,candidates):
 hits=[i for i,x in enumerate(ids[:10],1) if x in relevant]
 ideal=sum(1/math.log2(i+1) for i in range(1,min(10,len(relevant))+1))
 return {'top1':float(bool(ids) and ids[0] in relevant),'mrr10':1/hits[0] if hits else 0,'ndcg10':sum(1/math.log2(i+1) for i in hits)/ideal,'recall10':len(set(ids[:10])&relevant)/len(relevant),'recall50':len(set(ids[:50])&relevant)/len(relevant),'hit10':float(bool(hits)),'candidate_recall50':len(set(candidates[:50])&relevant)/len(relevant)}
def percentile(values,p):
 a=sorted(values);pos=(len(a)-1)*p;i=int(pos);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(pos-i)
def main():
 hashes=read(ROOT/'data/SHA256SUMS.json')
 for path,sha in hashes.items():require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,'File hash: '+path)
 with zipfile.ZipFile(ROOT/'downloads/mlqa-materials.zip') as z:
  data={c:json.loads(z.read('selected/'+c+'.json')) for c in ['eng-zho','zho-zho','eng-eng']}
  manifest=json.loads(z.read('dataset-manifest.json'))
  for c,entry in manifest['configs'].items():
   for item in entry['files'].values():require(hashlib.sha256(z.read(item['local_file'])).hexdigest()==item['sha256'],'Upstream data hash')
 with zipfile.ZipFile(ROOT/'downloads/results-v3.zip') as z:
  rows=[json.loads(x) for x in z.read('query-results.jsonl').splitlines() if x.strip()]
  ranks=[json.loads(x) for x in z.read('standalone-rankings.jsonl').splitlines() if x.strip()]
 cache={(r['config'],r['embedding'],r['query_id']):r['ids'] for r in ranks}
 require(len(rows)==6600 and len(cache)==2100,'Row counts')
 require(len({tuple(r[k] for k in ['config','embedding','reranker','query_id']) for r in rows})==6600,'Unique results')
 require(len({(r['config'],r['query_id']) for r in rows})==300,'Unique direction/query tasks')
 groups=collections.defaultdict(list)
 for r in rows:
  require(r['status']=='ok','Query error')
  candidates=cache[(r['config'],r['embedding'],r['query_id'])]
  final=candidates if r['reranker']=='standalone' else r['final_ids']
  require(r['reranker']=='standalone' or len(final)==50,'Rerank length')
  if r['reranker']!='standalone':require(set(final)==set(candidates[:50]),'Candidate set changed')
  rel=set(data[r['config']]['relevant'][r['query_id']])
  for k,v in metrics(final,rel,candidates).items():require(abs(v-r[k])<1e-9,'Metric mismatch '+str((r['config'],r['embedding'],r['query_id'],k)))
  groups[(r['config'],r['embedding'],r['reranker'])].append(r)
 require(len(groups)==66 and all(len(rs)==100 for rs in groups.values()),'66x100 matrix')
 a=read(ROOT/'data/analysis.json')
 for s in a['directions']:
  rs=groups[(s['config'],s['embedding'],s['reranker'])]
  for k in ['top1','mrr10','ndcg10','recall10','recall50']:require(abs(statistics.mean(r[k] for r in rs)-s[k])<1e-9,'Summary metric')
  for k in ['embedding_ms','retrieval_ms','rerank_ms','total_ms']:
   v=[r[k] for r in rs if r[k] is not None]
   for suffix,value in [('mean',statistics.mean(v) if v else None),('p50',percentile(v,.5) if v else None),('p95',percentile(v,.95) if v else None)]:
    actual=s[k+'_'+suffix];require(actual==value if value is None else abs(actual-value)<1e-7,'Timing mismatch')
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[];self.ids=set();self.title=False
  def handle_starttag(self,tag,attrs):
   d=dict(attrs)
   if 'id'in d:self.ids.add(d['id'])
   if tag=='title':self.title=True
   if tag in ['a','link','script','img']:
    self.links.extend(d[k] for k in ['href','src'] if k in d)
 docs={}
 for p in ROOT.rglob('*.html'):
  parser=Links();parser.feed(p.read_text(encoding='utf-8'));docs[p.resolve()]=parser
  require(parser.title,'Missing title')
 for p,parser in docs.items():
  for url in parser.links:
   u=urlsplit(url)
   if u.scheme or u.netloc:continue
   target=(p.parent/unquote(u.path)).resolve() if u.path else p
   require(target.exists(),f'Broken link {p.name}: {url}')
   if u.fragment and target in docs:require(u.fragment in docs[target].ids,f'Broken anchor {p.name}: {url}')
 print(json.dumps({'status':'passed','hashes':len(hashes),'ranking_metrics_recomputed':6600,'groups':66,'unique_direction_query_tasks':300,'upstream_parquet_hashes':9,'html_pages':len(docs),'scope':'offline integrity, metric recomputation and local links; not fresh model inference'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
