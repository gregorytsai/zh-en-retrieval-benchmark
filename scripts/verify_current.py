"""Independent ACORD metric and six-task publication check. No inference.

For full candidate-generation/request checks, extract the retest ZIP and run
python -m src.independent_validate . (the included standard-library validator).
"""
from pathlib import Path
import json,zipfile,math,statistics,collections,re,csv,hashlib
R=Path(__file__).resolve().parents[1]
def close(x,y):assert (x is None and y is None) or (x is not None and y is not None and abs(x-y)<1e-9),(x,y)
def verify():
 with zipfile.ZipFile(R/'downloads/acord-retest-20261027.zip') as z:
  for line in z.read('SHA256SUMS.txt').decode().splitlines():
   sha,path=line.split(maxsplit=1);assert hashlib.sha256(z.read(path.lstrip('*'))).hexdigest()==sha,path
  d=json.loads(z.read('frozen-data/acord.json'));rows=[json.loads(x) for x in z.read('query-results.jsonl').splitlines()];summ=json.loads(z.read('summary.json'))
  candidates={(r['embedder'],r['query_id']):r for r in map(json.loads,z.read('candidates.jsonl').splitlines())}
  scores=collections.defaultdict(list)
  for x in map(json.loads,z.read('reranker-scores.jsonl').splitlines()):scores[x['config_id'],x['query_id']].append(x)
 assert len(rows)==1596 and len({(r['config_id'],r['query_id']) for r in rows})==1596
 assert len(candidates)==399 and sum(map(len,scores.values()))==59850
 grouped=collections.defaultdict(list)
 for r in rows:
  assert r['status']=='ok' and not r['truncated']
  q=d['qrels'][r['query_id']];ids=r['final_doc_ids'];c=candidates[r['embedder'],r['query_id']]
  assert len(c['candidate_ids'])==len(set(c['candidate_ids']))==50
  assert set(c['candidate_ids'])<=set(q) and r['candidate_sha256']==c['ordered_candidate_sha256']
  if r['reranker']!='standalone':
   pairs=scores[r['config_id'],r['query_id']];assert len(pairs)==50
   assert [x['doc_id'] for x in sorted(pairs,key=lambda x:x['candidate_index'])]==c['candidate_ids']
   assert ids==[x['doc_id'] for x in sorted(pairs,key=lambda x:(-x['score'],x['doc_id']))]
  else:assert ids[:50]==c['candidate_ids']
  assert len(ids)==len(set(ids)) and set(ids)<=set(q)
  grades=[q[x] for x in ids];ideal=sorted(q.values(),reverse=True)
  dcg=lambda arr,k:sum(g/math.log2(i+2) for i,g in enumerate(arr[:k]))
  m=r['metrics']
  for k in [5,10]:close(m[f'ndcg@{k}'],dcg(grades,k)/dcg(ideal,k))
  positives={x for x,g in q.items() if g>=2};hits=[i for i,g in enumerate(grades[:10],1) if g>=2]
  close(m['top1'],float(grades[0]>=2));close(m['mrr@10'],1/hits[0] if hits else 0)
  close(m['hit@10'],float(bool(hits)));close(m['recall@10'],len(set(ids[:10])&positives)/len(positives));close(m['candidate_recall@50'],len(set(c['candidate_ids'])&positives)/len(positives))
  for g in (2,3,4):
   n=sum(x>=g for x in q.values());close(m[f'normalized_p5_grade_ge_{g}'],sum(x>=g for x in grades[:5])/min(n,5) if n else None)
  tm=r['times_ms'];close(tm['total'],sum(tm[k] for k in ['query_embedding','retrieval','evaluation_filter','rerank']))
  grouped[r['config_id']].append(r)
 for s in summ:
  rs=grouped[s['config_id']];assert len(rs)==57
  for k,v in s['metrics'].items():
   vals=[r['metrics'][k] for r in rs if r['metrics'][k] is not None];assert len(vals)==v['valid_n'];close(statistics.mean(vals),v['mean'])
  close(statistics.mean(r['times_ms']['total'] for r in rs),s['times_ms']['total']['mean'])
 a=json.loads((R/'data/matrix/analysis.json').read_text('utf8'));current=json.loads((R/'data/matrix/current-comparison.json').read_text('utf8'))
 legal=json.loads((R/'data/acord/retest.json').read_text('utf8'))['rows']
 mapping={'ndcg10':'ndcg@10','top1':'top1','mrr10':'mrr@10','recall50':'candidate_recall@50'}
 for r in legal:
  s=next(s for s in summ if s['embedder']==r['embedding'] and s['reranker']==r['reranker'])
  for k,v in mapping.items():close(r[k],s['metrics'][v]['mean'])
  for stat in ['mean','p50','p95']:close(r['total_ms_'+stat],s['times_ms']['total'][stat])
 source=[r for r in a['rows'] if r['dataset']!='acord-eng-eng']+legal
 for scope,rs in current['scopes'].items():
  assert len(rs)==28
  for r in rs:
   selected=[s for s in source if (s['embedding'],s['reranker'])==(r['embedding'],r['reranker']) and (True if scope=='all' else s['dataset']!='acord-eng-eng' if scope=='general' else s['dataset'] in ['eng-zho','zho-zho','eng-eng'] if scope=='mlqa' else s['dataset']==scope)]
   assert len(selected)==(6 if scope=='all' else 5 if scope=='general' else 3 if scope=='mlqa' else 1)
   for k in mapping:close(r[k],statistics.mean(s[k] for s in selected))
   close(r['total_ms_mean'],None if scope=='all' else statistics.mean(s['total_ms_mean'] for s in selected))
 for lang in ['', 'en/']:
  s=(R/(lang+'index.html')).read_text('utf8');payload=json.loads(re.search(r'<script id="phase-data" type="application/json">(.*?)</script>',s,re.S).group(1))
  assert payload['retest'] and payload['scopes']==current['scopes']
  assert len(re.findall(r'class="score-number"><strong>0\.\d{3}</strong>',s))==3
  assert len(re.findall('role="meter"',s))==3
 with (R/'data/matrix/current-summary.csv').open(encoding='utf-8-sig') as f:assert len(list(csv.DictReader(f)))==168
 return {'status':'passed','new_acord_rows':1596,'pair_scores':59850,'candidate_lists':399,'configurations':28,'current_task_rows':168,'six_task_equal_weight':True,'five_task_preserved':True,'bilingual_payloads':2,'new_inference':False}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
