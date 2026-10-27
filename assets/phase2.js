'use strict';
const PHASE=JSON.parse(document.querySelector('#phase-data').textContent);
const EN=PHASE.lang===1, say=(zh,en)=>EN?en:zh;
const scope=document.querySelector('#scope'),metric=document.querySelector('#metric'),reranker=document.querySelector('#reranker');
let sortKey=metric.value,sortDescending=true,expanded=false;
const label=k=>k==='standalone'?say('不加重排器','No reranker'):PHASE.names[k];
const title=r=>label(r.embedding)+(r.reranker==='standalone'?say('（單獨）',' (alone)'):' + '+label(r.reranker));
const score=(x,k)=>k==='top1'||k==='recall50'||k==='judged_coverage50'?(100*x).toFixed(2)+'%':x.toFixed(4);
const time=x=>x<100?x.toFixed(2)+' ms':(x/1000).toFixed(2)+' s';
const cols=[['embedding',say('Embedder','Embedder')],['reranker',say('重排器','Reranker')],['top1','Top-1'],['ndcg10','nDCG@10'],['mrr10','MRR@10'],['recall50','Recall@50'],['total_ms_mean',say('平均時間','Mean time')],['total_ms_p50','p50'],['total_ms_p95','p95'],['weight_gib_sum',say('權重 GiB','Weights GiB')]];
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function cellValue(r,key){if(key==='embedding'||key==='reranker')return label(r[key]);if(r[key]===null)return '—';if(key.startsWith('total_ms'))return time(r[key]);if(key==='weight_gib_sum')return r[key].toFixed(3);return score(r[key],key);}
function render(){
 const rows=PHASE.scopes[scope.value],filtered=rows.filter(r=>reranker.value==='all'||r.reranker===reranker.value);
 const macro=scope.value==='all'||scope.value==='mlqa';
 const info=macro?(scope.value==='all'?say('六分項各占 1/6。MLQA 占總權重 50%、MIRACL 33.33%、ACORD 16.67%。','Six conditions each weigh 1/6. MLQA weighs 50%, MIRACL 33.33%, and ACORD 16.67%.'):say('MLQA 三方向各占 1/3。','Three MLQA directions each weigh 1/3.'))+say(' 完整受測組合：',' Complete setups: ')+rows.length:PHASE.labels[scope.value]+' · '+PHASE.datasets[scope.value].queries+say(' 題 · ',' queries · ')+PHASE.datasets[scope.value].corpus.toLocaleString()+say(' 篇文件',' documents');
 document.querySelector('#scope-note').textContent=info+(scope.value==='acord-eng-eng'?say(' Top-1 採本測試門檻：2 星以上；nDCG 使用指數 gain，與 ACORD 原論文指標設定不同。',' Top-1 uses this study’s threshold of at least 2 stars; nDCG uses exponential gain, unlike the original ACORD protocol.'):'')+' '+say('顯示 ','Showing ')+filtered.length+say(' 種組合。',' setups.');
 const bars=document.querySelector('#phase-bars');bars.replaceChildren();
 if(!filtered.length)bars.append(el('p',say('此範圍沒有這個重排器的完整測量；請切換分項或重排器。','No complete measurement for this reranker in this scope. Change the task or reranker.'),'note'));
 const toggle=document.querySelector('#toggle-bars'),ordered=[...filtered].sort((a,b)=>b[metric.value]-a[metric.value]);
 const chartRows=toggle&&!expanded?ordered.slice(0,7):ordered;
 if(toggle){toggle.hidden=filtered.length<=7;toggle.textContent=expanded?say('收合為前7名','Show top seven'):say('目前顯示前7名；展開全部 ','Showing top seven; show all ')+filtered.length+say(' 組合',' setups');toggle.setAttribute('aria-expanded',String(expanded));}
 for(const r of chartRows){
  const base=rows.find(b=>b.embedding===r.embedding&&b.reranker==='standalone'),v=r[metric.value],b=base[metric.value],delta=v-b;
  const row=el('div',undefined,'phase-bar-row');row.dataset.setup=r.embedding+'+'+r.reranker;row.dataset.value=String(v);row.dataset.baseline=String(b);
  const text=el('div',undefined,'phase-bar-label');text.append(el('strong',title(r)),el('small',r.weight_gib_sum.toFixed(3)+' GiB · '+time(r.total_ms_mean)+say(' 平均',' mean')));
  const track=el('div',undefined,'phase-track');track.setAttribute('role','img');track.setAttribute('aria-label',title(r)+' '+metric.value+' '+score(v,metric.value)+', '+say('單獨檢索 ','baseline ')+score(b,metric.value));
  const fill=el('span',undefined,'phase-base');fill.style.width=Math.min(v,b)*100+'%';track.append(fill);
  if(delta!==0){const extra=el('span',undefined,delta>0?'phase-gain':'phase-loss');extra.style.left=Math.min(v,b)*100+'%';extra.style.width=Math.abs(delta)*100+'%';track.append(extra);}
  const deltaLabel=['ndcg10','mrr10'].includes(metric.value)?(delta>=0?'+':'')+delta.toFixed(4):(delta>=0?'+':'')+(delta*100).toFixed(2)+' pp';
  const val=el('div',undefined,'phase-bar-value');val.append(el('strong',score(v,metric.value)),el('small',r.reranker==='standalone'?say('基準','Baseline'):deltaLabel));
  row.append(text,track,val);bars.append(row);
 }
 const table=el('table');table.setAttribute('aria-label',say('最新結果比較','Latest result comparison'));const head=el('thead'),hr=el('tr');
 for(const [key,text] of cols){const th=el('th');th.scope='col';th.setAttribute('aria-sort',sortKey===key?(sortDescending?'descending':'ascending'):'none');const button=el('button',text+(sortKey===key?(sortDescending?' ↓':' ↑'):' ↕'));button.type='button';button.dataset.sort=key;button.addEventListener('click',()=>{sortDescending=sortKey===key?!sortDescending:!['embedding','reranker','weight_gib_sum','total_ms_mean','total_ms_p50','total_ms_p95'].includes(key);sortKey=key;render();});th.append(button);hr.append(th);}
 head.append(hr);table.append(head);const body=el('tbody');
 const sorted=[...filtered].sort((a,b)=>{if(a[sortKey]===null||b[sortKey]===null)return 0;const x=typeof a[sortKey]==='string'?label(a[sortKey]).localeCompare(label(b[sortKey])):a[sortKey]-b[sortKey];return sortDescending?-x:x;});
 for(const r of sorted){const tr=el('tr');tr.dataset.setup=r.embedding+'+'+r.reranker;tr.dataset.sortValue=String(r[sortKey]);for(const [key] of cols){const td=el('td');if(['embedding','reranker'].includes(key)&&r[key]!=='standalone'){const a=el('a',label(r[key]));a.href='models/'+r[key]+'.html';td.append(a);}else td.textContent=cellValue(r,key);tr.append(td);}body.append(tr);}table.append(body);document.querySelector('#phase-table').replaceChildren(table);
}
document.querySelector('#toggle-bars')?.addEventListener('click',()=>{expanded=!expanded;render();});
for(const node of [scope,metric,reranker])node.addEventListener('change',()=>{if(node===metric){sortKey=metric.value;sortDescending=true;}render();});render();
