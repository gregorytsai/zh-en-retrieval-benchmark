"""Package source and hash the reviewable public artifacts (no model inference)."""
from pathlib import Path
import zipfile,json,hashlib,io
ROOT=Path(__file__).resolve().parents[1]
for p in ROOT.rglob('*'):
 if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix!='.zip':
  raw=p.read_bytes();p.write_bytes(raw.replace(b'\r\n',b'\n'))
with zipfile.ZipFile(ROOT/'downloads/benchmark-code.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted((ROOT/'research/workspace').rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT/'research'))
 for name in ['LICENSE','research/BUILD_GUIDE_ZH_TW.md','research/PUBLICATION_CHANGES.md']:z.write(ROOT/name,Path(name).name)
with zipfile.ZipFile(ROOT/'downloads/phase2-code.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted((ROOT/'research/phase2').rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
 z.write(ROOT/'scripts/verify_phase2.py','scripts/verify_phase2.py')
 z.write(ROOT/'LICENSE','LICENSE')
for filename in ['phase2-batch1-results.zip','phase2-batch2-results.zip','phase2-materials.zip']:
 path=ROOT/'downloads'/filename
 with zipfile.ZipFile(path) as z:entries={n:z.read(n) for n in z.namelist()}
 entries['LICENSES/Apache-2.0.txt']=(ROOT/'research/phase2/LICENSES/Apache-2.0.txt').read_bytes()
 out=io.BytesIO()
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,data in entries.items():
   info=zipfile.ZipInfo(n,(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
 path.write_bytes(out.getvalue())
publication_path=ROOT/'research/matrix/PUBLICATION_MANIFEST.json'
publication=json.loads(publication_path.read_text('utf-8'))
for relative,metadata in publication.items():
 metadata['published_sha256']=hashlib.sha256((ROOT/'research/matrix/workspace/embedding-study-phase2-20261023'/relative).read_bytes()).hexdigest()
publication_path.write_bytes((json.dumps(publication,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
with zipfile.ZipFile(ROOT/'downloads/matrix-code.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted((ROOT/'research/matrix').rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
 for name in ['scripts/verify_matrix.py','scripts/verify_correction.py','scripts/compare_qwen_bge.py','scripts/recompute_acord.py','data/matrix/analysis.json','data/matrix/current-comparison.json','data/matrix/acord-corrected.json','data/matrix/provenance.json','research/phase2/LICENSES/Apache-2.0.txt','LICENSE']:z.write(ROOT/name,name)
hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.name!='SHA256SUMS.json'}
(ROOT/'data/SHA256SUMS.json').write_bytes((json.dumps(hashes,indent=2)+'\n').encode('utf-8'))
print('Packaged code and hashed',len(hashes),'files')
