"""Verify corrected metrics, five-task payloads and all recorded timing sums.

Standard-library only. No inference or network. Does not reproduce vector scores;
cache SHA checks and original Top50 parity are recorded by recompute_acord.py.
"""
from pathlib import Path
import json,zipfile,math,statistics,re,collections,csv
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text('utf-8'))
def close(a,b):assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-10),(a,b)
def verify():
 with zipfile.ZipFile(R/'downloads/matrix-shared.zip') as z:j=json.loads(z.read('qrels.json'))['datasets']['acord-eng-eng']['qrels']
 with zipfile.ZipFile(R/'downloads/acord-corrected-rankings.zip') as z:rows=[json.loads(l) for l in z.read('judged-rankings.jsonl').splitlines()]
 assert len(rows)==399 and len({(r['embedding'],r['query_id']) for r in rows})==399
 groups=collections.defaultdict(list)
 for r in rows:
  grades=j[r['query_id']];ids=r['judged_ids'];assert len(ids)==len(grades) and set(ids)==set(grades)
  actual=[grades[d] for d in ids];ideal=sorted(grades.values(),reverse=True)
  dcg=lambda v:sum(v[i]/math.log2(i+2) for i in range(min(10,len(v))))
  close(r['ndcg10'],dcg(actual)/dcg(ideal))
  positives={d for d,g in grades.items() if g>=2};hits=[i for i,g in enumerate(actual[:10],1) if g>=2]
  close(r['top1'],float(actual[0]>=2));close(r['mrr10'],1/hits[0] if hits else 0)
  close(r['recall50'],len(set(ids[:50])&positives)/len(positives))
  for stars in [3,4,5]:
   total=sum(g>=stars-1 for g in grades.values());expected=sum(g>=stars-1 for g in actual[:5])/min(5,total) if total else None
   close(r[f'p5_stars{stars}'],expected)
  groups[r['embedding']].append(r)
 corrected=read('data/matrix/acord-corrected.json')
 for s in corrected['rows']:
  rs=groups[s['embedding']];assert len(rs)==57
  for field in ['ndcg10','top1','mrr10','recall50','p5_stars3','p5_stars4','p5_stars5']:close(s[field],statistics.mean(r[field] for r in rs if r[field] is not None))
  assert all(s[f] is None for f in ['total_ms_mean','total_ms_p50','total_ms_p95'])
 a=read('data/matrix/analysis.json');current=read('data/matrix/current-comparison.json');assert len(current['scopes']['acord-eng-eng'])==7
 for scope,rs in current['scopes'].items():
  if scope=='acord-eng-eng':assert rs==corrected['rows'];continue
  assert len(rs)==28
  for r in rs:
   selected=[s for s in a['rows'] if s['embedding']==r['embedding'] and s['reranker']==r['reranker'] and (s['dataset']!='acord-eng-eng' if scope=='all' else s['dataset'] in ['eng-zho','zho-zho','eng-eng'] if scope=='mlqa' else s['dataset']==scope)]
   assert len(selected)==(5 if scope=='all' else 3 if scope=='mlqa' else 1)
   for f in ['ndcg10','top1','mrr10','recall50','total_ms_mean']:close(r[f],statistics.mean(s[f] for s in selected))
 count=0;diaggroups=collections.defaultdict(list)
 for i in range(1,8):
  with zipfile.ZipFile(R/f'downloads/matrix-B{i:02}-results.zip') as z:
   for line in z.read('query-results.jsonl').splitlines():
    r=json.loads(line);close(r['total_ms'],r['embedding_ms']+r['retrieval_ms']+(r['rerank_ms'] or 0));count+=1
    if r['dataset']=='acord-eng-eng':diaggroups[r['embedding'],r['reranker']].append(r)
 for diag in read('data/matrix/acord-diagnostic.json')['results']:
  rs=diaggroups[diag['embedding'],diag['reranker']];vals=[]
  for r in rs:
   grades=j[r['query_id']];ids=[d for d in r['final_ids'] if d in grades]
   dcg=lambda v:sum(g/math.log2(i+2) for i,g in enumerate(v[:10]))
   vals.append(dcg([grades[d] for d in ids])/dcg(sorted(grades.values(),reverse=True)))
  close(diag['condensed_saved50_linear_ndcg'],statistics.mean(vals))
 for lang in ['', 'en/']:
  s=(R/(lang+'index.html')).read_text('utf-8');payload=json.loads(re.search(r'<script id="phase-data" type="application/json">(.*?)</script>',s,re.S).group(1))
  assert payload['corrected'] and payload['scopes']==current['scopes']
  assert len(re.findall('class="score-meter"',s))==3 and len(re.findall(r'class="score-number"><strong>0\.\d{3}</strong>',s))==3
  assert 'role="meter"' in s and 'aria-valuemax="1"' in s
  for p in (R/(lang+'models')).glob('*.html'):
   txt=p.read_text('utf-8');assert ('acord.html' in txt) and ('ACORD' in txt)
 with (R/'data/matrix/current-summary.csv').open(encoding='utf-8-sig') as f:assert len(list(csv.DictReader(f)))==147
 assert count==73164
 return {'status':'passed','corrected_rankings':399,'corrected_models':7,'five_task_configurations':28,'timing_sums_verified':count,'diagnostic_configurations':28,'bilingual_payloads_verified':2,'new_inference':False}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
