"""Rebuild ACORD judged-only rankings from verified original NumPy caches.

Usage: python scripts/recompute_acord.py CACHE_ROOT
Each CACHE_ROOT/model/ contains corpus.npy, queries.npy, candidates.json and
cache-manifest.json. No inference. Public rank evidence supports metric checks
without redistributing the vector caches. See verify_correction.py.
"""
from pathlib import Path
import sys,json,zipfile,hashlib,statistics,math,csv,io
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def metrics(ids,j):
    assert set(ids)==set(j) and len(ids)==len(j)
    def dcg(grades):return sum(g/math.log2(i+2) for i,g in enumerate(grades[:10]))
    positive={d for d,g in j.items() if g>=2} # original stars >= 3
    hits=[i for i,d in enumerate(ids[:10],1) if d in positive]
    result={'ndcg10':dcg([j[d] for d in ids])/dcg(sorted(j.values(),reverse=True)),
        'top1':float(ids[0] in positive),'mrr10':1/hits[0] if hits else 0.,
        'recall50':len(set(ids[:50])&positive)/len(positive),'judged_coverage50':1.0}
    for stars in [3,4,5]:
        relevant={d for d,g in j.items() if g>=stars-1}
        result[f'p5_stars{stars}']=sum(d in relevant for d in ids[:5])/min(5,len(relevant)) if relevant else None
    return result

def main(cache):
    with zipfile.ZipFile(ROOT/'downloads/phase2-materials.zip') as z:
        d=json.loads(z.read('batch1/datasets.json'))['datasets']['acord-eng-eng']
    with zipfile.ZipFile(ROOT/'downloads/matrix-shared.zip') as z:
        qrels=json.loads(z.read('qrels.json'))['datasets']['acord-eng-eng']['qrels']
    assert d['qrels']==qrels
    a=json.loads((ROOT/'data/matrix/analysis.json').read_text('utf-8'))
    docs=np.asarray([x['id'] for x in d['corpus']],dtype=str)
    evidence=[];summaries=[];provenance={};original_rank_matches=0
    models=['qwen-4b','qwen-8b','embeddinggemma-2','qwen-0.6b','bge-m3','f2llm-v2-0.6b','granite-97m-r2']
    for batch,e in enumerate(models,1):
        folder=cache/e; m=json.loads((folder/'cache-manifest.json').read_text('utf-8'))
        assert m['embedding']==e and m['corpus_documents']==len(docs)==3931
        assert m['model_sha256']==a['models'][e]['sha256']
        assert len(m['query_ids'])==57 and set(m['query_ids'])==set(qrels)
        for n,h in m['files_sha256'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==h
        corpus=np.load(folder/'corpus.npy');queries=np.load(folder/'queries.npy')
        assert corpus.shape==(3931,m['dimension']) and queries.shape==(57,m['dimension'])
        assert np.isfinite(corpus).all() and np.isfinite(queries).all()
        assert np.max(np.abs(np.linalg.norm(corpus,axis=1)-1))<1e-5
        with zipfile.ZipFile(ROOT/f'downloads/matrix-B{batch:02}-results.zip') as z:
            raw=[json.loads(l) for l in z.read('query-results.jsonl').splitlines()]
        saved={r['query_id']:r for r in raw if r['dataset']=='acord-eng-eng' and r['reranker']=='standalone'}
        rows=[]
        for qid,v in zip(m['query_ids'],queries):
            scores=corpus@v;indices=np.lexsort((docs,-scores));full=docs[indices].tolist()
            assert full[:50]==saved[qid]['final_ids'],(e,qid,'cached rank mismatch')
            original_rank_matches+=1
            ranked=[doc for doc in full if doc in qrels[qid]]
            result=metrics(ranked,qrels[qid]);rows.append(result)
            evidence.append({'embedding':e,'query_id':qid,'judged_ids':ranked,**result})
        summary={k:statistics.mean(r[k] for r in rows if r[k] is not None) for k in rows[0]}
        summary.update(embedding=e,reranker='standalone',queries=57,weight_gib_sum=a['models'][e]['weight_gib'],
            total_ms_mean=None,total_ms_p50=None,total_ms_p95=None,
            p5_stars5_evaluable_queries=sum(r['p5_stars5'] is not None for r in rows))
        summaries.append(summary)
        provenance[e]={'cache_manifest_sha256':hashlib.sha256((folder/'cache-manifest.json').read_bytes()).hexdigest(),
            'files_sha256':m['files_sha256'],'dataset_sha256':m['dataset_sha256'],'model_sha256':m['model_sha256'],
            'dimension':m['dimension'],'query_ids':m['query_ids'],'original_top50_rankings_matched':57}
    out={'version':'2026-10-27-correction','protocol':'Filter full corpus rankings to explicitly judged query-document pairs before cutoffs; linear grade gain; binary metrics use original stars >=3. Star P@5 normalizes by min(5, viable count); zero-viable queries excluded.',
        'scope':'Seven standalone embedders only; not a reranker rerun or a full reproduction of paper model configurations.',
        'timing':'Corrected standalone timings are not remeasured; not shown alongside the new quality protocol.',
        'rows':sorted(summaries,key=lambda r:-r['ndcg10']),'provenance':provenance,
        'verification':{'original_top50_rankings_matched':original_rank_matches,'queries':399,'new_inference':False}}
    (ROOT/'data/matrix/acord-corrected.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'downloads/acord-corrected-rankings.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        z.writestr('judged-rankings.jsonl','\n'.join(json.dumps(r) for r in evidence)+'\n')
        z.writestr('NOTICE.txt','ACORD: The Atticus Project and authors, CC BY 4.0. https://huggingface.co/datasets/theatticusproject/acord . Derived full judged-only rankings from original saved vectors; no text, vectors or model weights included. No new inference. Recompute metrics with scripts/verify_correction.py.\n')
    print(json.dumps({'verification':out['verification'],'scores':[(r['embedding'],r['ndcg10']) for r in out['rows']]},indent=2))

if __name__=='__main__':main(Path(sys.argv[1]))
