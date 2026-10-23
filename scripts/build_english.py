"""Build complete English pages from shared markup and reviewed translations."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,urlunsplit
import html,json,posixpath,re
ROOT=Path(__file__).resolve().parents[1]
SITE='https://gregorytsai.github.io/zh-en-retrieval-benchmark/'
T=json.loads((ROOT/'scripts/en_translations.json').read_text(encoding='utf-8'))
CJK=re.compile('[\u3400-\u9fff]')
def translate(value):
 stripped=value.strip()
 if stripped in T:return value[:len(value)-len(value.lstrip())]+T[stripped]+value[len(value.rstrip()):]
 if stripped=='、':return ' and '
 if not CJK.search(value):return value
 if stripped.startswith('{'):
  try:
   obj=json.loads(stripped)
   return json.dumps({k:translate(v) if isinstance(v,str) else v for k,v in obj.items()},ensure_ascii=False,indent=2)
  except json.JSONDecodeError:pass
 result=value
 for source in sorted(T,key=len,reverse=True):result=result.replace(source,T[source])
 if CJK.search(result):raise ValueError('Missing English translation: '+result)
 return result

class EnglishPage(HTMLParser):
 def __init__(self,path,pages):
  super().__init__(convert_charrefs=True);self.path=path;self.pages=pages;self.out=[]
 def handle_decl(self,decl):self.out.append('<!'+decl+'>')
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if tag=='html':attrs['lang']='en'
  for k in ['content','aria-label','title','placeholder','data-label']:
   if attrs.get(k):attrs[k]=translate(attrs[k])
  if 'data-sort-values' in attrs:
   attrs['data-sort-values']=json.dumps([translate(v) if isinstance(v,str) else v for v in json.loads(attrs['data-sort-values'])],ensure_ascii=False)
  for k in ['href','src']:
   if not attrs.get(k):continue
   u=urlsplit(attrs[k])
   if tag=='link' and attrs.get('rel')=='canonical':attrs[k]=SITE+'en/'+self.path;continue
   if u.scheme or u.netloc or not u.path:continue
   target=posixpath.normpath(posixpath.join(posixpath.dirname(self.path),u.path))
   # Localized HTML and app.js retain the same structure below en/.
   if target in self.pages or target=='assets/app.js':continue
   attrs[k]=urlunsplit(('', '',posixpath.relpath(target,posixpath.dirname('en/'+self.path)),u.query,u.fragment))
  self.out.append('<'+tag+''.join(' '+k if v is None else ' '+k+'="'+html.escape(v,quote=True)+'"' for k,v in attrs.items())+'>')
 def handle_endtag(self,tag):self.out.append('</'+tag+'>')
 def handle_data(self,data):self.out.append(html.escape(translate(data),quote=False))
 def handle_comment(self,data):self.out.append('<!--'+data+'-->')

def build(pages):
 (ROOT/'en/assets').mkdir(parents=True,exist_ok=True)
 for path in pages:
  original=(ROOT/path).read_text(encoding='utf-8')
  alternates=f'<link rel="alternate" hreflang="zh-Hant" href="{SITE+path}"><link rel="alternate" hreflang="en" href="{SITE}en/{path}">'
  css=posixpath.relpath('assets/i18n.css',posixpath.dirname(path) or '.')
  to_en=posixpath.relpath('en/'+path,posixpath.dirname(path) or '.')
  zh=original.replace('</nav>',f'<a class="language-switch" lang="en" hreflang="en" href="{to_en}">English</a></nav>').replace('</head>',alternates+f'<link rel="stylesheet" href="{css}"></head>')
  (ROOT/path).write_text(zh,encoding='utf-8')
  parser=EnglishPage(path,set(pages));parser.feed(original)
  to_zh=posixpath.relpath(path,posixpath.dirname('en/'+path))
  css=posixpath.relpath('assets/i18n.css',posixpath.dirname('en/'+path))
  en=''.join(parser.out).replace('</nav>',f'<a class="language-switch" lang="zh-Hant" hreflang="zh-Hant" href="{to_zh}">繁體中文</a></nav>').replace('</head>',alternates+f'<link rel="stylesheet" href="{css}"></head>')
  destination=ROOT/'en'/path;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(en,encoding='utf-8')
 data,js=(ROOT/'assets/app.js').read_text(encoding='utf-8').split(';\n',1)
 data=json.loads(data.removeprefix('const BENCHMARK='));data['names']={k:translate(v) for k,v in data['names'].items()}
 ui={
  '此預算下沒有已測組合。':'No tested configuration fits this weight budget.',
  '檢閱模型與已測配對 →':'View model and tested pairings →',
  '顯示 ${count} / 22 種組合':'Showing ${count} of 22 configurations',
  '排序：${button.dataset.label}，由${sortDescending ? \'高到低\' : \'低到高\'}':"Sorted by ${button.dataset.label}, ${sortDescending ? 'highest first' : 'lowest first'}",
  '三方向等權平均，每個已測組合 300 題':'three directions equally weighted; 300 tasks per measured configuration',
  '每個已測組合 100 題':'100 tasks per measured configuration',
  '；Granite 與 F2LLM 沒有重排測量，灰色長條僅供單模型參考':'; Granite and F2LLM have no reranking measurements. Gray bars show standalone results only',
  '按${hasReranker ? \'重排後\' : \'單模型\'}命中率由高到低排列':"Sorted by ${hasReranker ? 'reranked' : 'standalone'} Top-1, highest first",
  '加上重排器，命中率提升多少？':'How much does reranking improve retrieval?',
  '未測此配對':'Pair not tested','單模型':'Standalone','原本':'Base',
  '權重合計':'Weights','本機平均':'Mean on test machine','平均 Top-1':'Mean Top-1',
  '百分點':'pp','；':'; ', '。':'. ',"'zh-Hant'":"'en'"
 }
 # Translate full dynamic sentences before smaller labels to preserve grammar.
 for source in sorted(ui,key=len,reverse=True):js=js.replace(source,ui[source])
 for source in sorted(T,key=len,reverse=True):js=js.replace(source,T[source].replace("'","\\'"))
 if CJK.search(js):raise ValueError('Untranslated UI: '+str(re.findall('.{0,45}[\u3400-\u9fff].{0,60}',js)))
 (ROOT/'en/assets/app.js').write_text('const BENCHMARK='+json.dumps(data,ensure_ascii=False)+';\n'+js,encoding='utf-8')
 all_pages=pages+['en/'+p for p in pages]
 (ROOT/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+''.join('<url><loc>'+SITE+p+'</loc></url>' for p in all_pages)+'</urlset>',encoding='utf-8')
 print('Built',len(pages),'English pages and bilingual navigation')
