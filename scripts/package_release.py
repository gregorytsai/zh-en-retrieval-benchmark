"""Package source and hash the reviewable public artifacts (no model inference)."""
from pathlib import Path
import zipfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(ROOT/'downloads/benchmark-code.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted((ROOT/'research/workspace').rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT/'research'))
 for name in ['LICENSE','research/BUILD_GUIDE_ZH_TW.md','research/PUBLICATION_CHANGES.md']:z.write(ROOT/name,Path(name).name)
hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.name!='SHA256SUMS.json'}
(ROOT/'data/SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
print('Packaged code and hashed',len(hashes),'files')
