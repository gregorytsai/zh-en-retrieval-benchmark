"""Rebuild frozen MIRACL inputs from public selection IDs and pinned upstream passages.

Downloads ~115 MB on first use. No model inference or remote code execution.
Run in the repository: python research/phase2/fetch_miracl_materials.py OUTPUT_DIRECTORY
Retain MIRACL / Wikipedia attribution and applicable upstream text terms.
"""
from pathlib import Path
import gzip,hashlib,json,sys,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[2]
FILES=[('en.jsonl.gz','https://huggingface.co/datasets/miracl/nomiracl/resolve/ecd08778d0426a5ca28ac99763b0c9ddc2c78e68/data/english/corpus.jsonl.gz','f333c75491c8dac69f946b021511a67666318efb799dff3a6d927a841f8148eb'),('zh.jsonl.gz','https://huggingface.co/datasets/miracl/nomiracl/resolve/ecd08778d0426a5ca28ac99763b0c9ddc2c78e68/data/chinese/corpus.jsonl.gz','d09a33559280fe5e7b4a6ce9330ff9e261906d1ef4bf991ea8c02d309657f9fa'),('zh-original.jsonl.gz','https://huggingface.co/datasets/miracl/miracl-corpus/resolve/d921ec7e349ce0d28daf30b2da9da5ee698bef0d/miracl-corpus-v1.0-zh/docs-0.jsonl.gz','b036ba6a927a4598b314de55a1de200d2a619b28b4327f49e97ec4233d324827')]
EXPECTED='fdf802cc03e9059783e5a2cba0785437253f07841ec6956246d5884461e7ba3d'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rebuild(destination):
 destination.mkdir(parents=True,exist_ok=True)
 for filename,url,digest in FILES:
  p=destination/filename
  if not p.exists():
   print('Downloading',filename)
   with urllib.request.urlopen(url,timeout=120) as response,p.open('wb') as f:
    while chunk:=response.read(1024*1024):f.write(chunk)
  if sha(p)!=digest:raise RuntimeError('Upstream hash mismatch: '+filename)
 with zipfile.ZipFile(ROOT/'downloads/phase2-materials.zip') as z:data=json.loads(z.read('batch2/selection-without-passage-text.json'))
 for key,d in data['datasets'].items():
  lang=d['metadata']['language'];wanted={x['id'] for x in d['corpus']};docs={}
  for filename in [lang+'.jsonl.gz']+(['zh-original.jsonl.gz'] if lang=='zh' else []):
   with gzip.open(destination/filename,'rt',encoding='utf-8') as f:
    for line in f:
     row=json.loads(line)
     if row['docid'] in wanted and row['docid'] not in docs:docs[row['docid']]=row
  if set(docs)!=wanted:raise RuntimeError('Missing passages')
  d['corpus']=[{'id':x['id'],'text':(docs[x['id']]['title']+'\n'+docs[x['id']]['text']).strip(),'metadata':x['metadata']} for x in d['corpus']]
 raw=(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
 if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise RuntimeError('Reconstructed dataset hash mismatch')
 path=destination/'datasets-batch2.json'
 if path.exists() and path.read_bytes()!=raw:raise RuntimeError('Refusing to overwrite a different dataset')
 path.write_bytes(raw);print('Verified original dataset SHA256:',EXPECTED)
if __name__=='__main__':rebuild(Path(sys.argv[1]))
