"""Feuille de style et scripts partagés du tableau de bord (générés par Python, sans framework)."""

from __future__ import annotations

import json

from pokerlab.report import theme
from pokerlab.report.html import CSS as REPORT_CSS

NAV_CSS = """
:root{--accent:#2a78d6;--good:#0ca30c;--bad:#d03b3b;--felt:#1d5c43;--felt-ink:#f0efec;--card:#fcfcfb;--card-ink:#0b0b0b;--card-red:#d03b3b}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])){--accent:#3987e5;--felt:#173f30;--card:#f0efec}}
:root[data-theme="dark"]{--accent:#3987e5;--felt:#173f30;--card:#f0efec}
header.site{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--ring)}
header.site .in{max-width:1100px;margin:0 auto;padding:8px 16px;display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px}
header.site b{font-size:1rem;margin-right:6px}
header.site a{color:var(--ink2);text-decoration:none;padding:4px 2px;font-size:.9rem}
header.site a.on{color:var(--ink);font-weight:700;border-bottom:2px solid var(--accent)}
header.site button.theme{float:none;margin-left:auto;padding:3px 8px;font-size:.8rem}
@media (max-width:420px){header.site .in{gap:4px 10px;padding:8px 12px}header.site b{display:none}}
"""

SITE_CSS = REPORT_CSS + NAV_CSS + """
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.kpi{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px}
.kpi .v{font-size:1.5rem;font-weight:700;color:var(--ink);line-height:1.2}
.kpi .l{font-size:.8rem;color:var(--muted)}
.banner{border-left:4px solid var(--accent);background:var(--surface);padding:10px 12px;border-radius:8px;margin:10px 0;color:var(--ink)}
.banner.warn{border-left-color:var(--bad)}.banner.ok{border-left-color:var(--good)}
.tag-ex{display:inline-block;background:#eda100;color:#0b0b0b;font-weight:700;border-radius:6px;padding:1px 8px;font-size:.8rem}
.progress{height:14px;background:var(--grid);border-radius:7px;overflow:hidden}
.progress>div{height:100%;background:var(--accent)}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.filters select,.filters input{font:inherit;font-size:.9rem;padding:6px 8px;border:1px solid var(--ring);border-radius:8px;background:var(--surface);color:var(--ink);max-width:100%}
.filters input{flex:1 1 160px}
table.hands tr{cursor:pointer}
@media (max-width:560px){table.hands th:nth-child(2),table.hands td:nth-child(2),table.hands th:nth-child(5),table.hands td:nth-child(5){display:none}}table.hands tr:hover td{background:var(--page)}
td.pos{color:var(--good)}td.neg{color:var(--bad)}
.felt{position:relative;width:100%;max-width:640px;aspect-ratio:16/11;margin:10px auto;background:var(--felt);border-radius:48%;
border:6px solid #6b4a2b;color:var(--felt-ink)}
.felt .center{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);text-align:center;width:60%}
.felt .pot{font-size:.85rem;margin-top:4px}
.seat{position:absolute;transform:translate(-50%,-50%);background:rgba(0,0,0,.55);border-radius:8px;padding:4px 6px;
min-width:74px;max-width:30%;text-align:center;font-size:.72rem;line-height:1.25;color:#f0efec}
.seat.act{outline:2px solid #eda100}.seat.out{opacity:.45}.seat.hero{background:rgba(42,120,214,.75)}
.seat .nm{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.seat .bub{display:block;margin-top:2px;color:#ffe29a;white-space:normal}
.cards{display:inline-flex;gap:2px;justify-content:center;flex-wrap:nowrap}
.c{display:inline-block;background:var(--card);color:var(--card-ink);border-radius:3px;padding:1px 2px;min-width:22px;font-weight:700;font-size:.8rem;
font-variant-numeric:normal;box-shadow:0 1px 1px rgba(0,0,0,.4)}
.c.r{color:var(--card-red)}.c.back{background:#2a78d6;color:#2a78d6}
.controls{display:flex;gap:8px;align-items:center;justify-content:center;margin:6px 0}
.controls button{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--ring);background:var(--surface);color:var(--ink)}
ol.steps{padding-left:2.2rem;max-height:340px;overflow:auto}
#replay{scroll-margin-top:64px}
ol.steps li{margin:.15rem 0;color:var(--ink2)}ol.steps li.cur{color:var(--ink);font-weight:700}
.ok{color:var(--good)}.ko{color:var(--bad)}
.sr{position:absolute;left:-9999px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:0 12px}
.cols h3{margin:.2rem 0 .4rem}
ul.camps{padding-left:1rem}ul.camps li{margin:.45rem 0;color:var(--ink)}
.tag{font-size:.75rem;color:var(--muted)}
td.l,th.l{text-align:left;white-space:normal}
.sw{flex:none}
"""

SITE_JS = """
const DARK_MAP = %(dark)s;
function isDark(){const t=document.documentElement.getAttribute('data-theme');
 if(t==='dark')return true; if(t==='light')return false; return window.matchMedia('(prefers-color-scheme: dark)').matches;}
function themed(txt){ if(!isDark()) return txt; return txt.replace(/#[0-9a-fA-F]{6}/g, m => DARK_MAP[m.toLowerCase()] || m); }
function renderFigures(){ if(!window.Plotly) return; document.querySelectorAll('script.figdata').forEach(s => {
  const fig = JSON.parse(themed(s.textContent)); const el = document.getElementById(s.dataset.target);
  if(el) Plotly.react(el, fig.data, fig.layout, {responsive:true, displaylogo:false}); }); }
function toggleTheme(){ const r=document.documentElement; r.setAttribute('data-theme', isDark()?'light':'dark'); renderFigures(); }
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderFigures);
document.addEventListener('DOMContentLoaded', renderFigures);
(function(){ if(!location.protocol.startsWith('http')) return;
  setInterval(async () => { try { const r = await fetch('version.json', {cache:'no-store'}); const v = (await r.json()).version;
    if (window.SITE_VERSION && v !== window.SITE_VERSION) location.reload(); } catch (e) {} }, 4000); })();
"""

MAINS_JS = r"""
(function(){
const HANDS = window.HANDS || [];
const REPLAYS = window.REPLAYS || {};
const $ = s => document.querySelector(s);
const STREETS = ['Préflop','Flop','Turn','River'];
function cardHtml(c, hidden){ if(hidden) return '<span class="c back">??</span>';
  const red = c[1]==='h'||c[1]==='d'; const sym = {s:'♠',h:'♥',d:'♦',c:'♣'}[c[1]];
  return '<span class="c'+(red?' r':'')+'">'+c[0].replace('T','10')+sym+'</span>'; }
function cards(list, hidden){ return '<span class="cards">'+list.map(c=>cardHtml(c, hidden)).join('')+'</span>'; }
function fmt(v){ return (v>0?'+':'')+v.toFixed(1).replace('.',','); }
let shown = 200;
function filtered(){
  const q = ($('#q').value||'').toLowerCase().replace(/\s/g,'');
  const se = $('#f-session').value, po = $('#f-pos').value, re = $('#f-res').value, hf = $('#f-hf').value;
  return HANDS.filter(h => (se==='' || String(h.session)===se) && (po==='' || h.position===po)
    && (re==='' || (re==='g' ? h.result>0 : h.result<0)) && (hf==='' || String(h.hors_fiche)===hf)
    && (q==='' || (h.cards+h.classe+h.id+h.board).toLowerCase().includes(q)));
}
function renderList(){
  const rows = filtered(); const body = $('#hands tbody'); body.innerHTML = '';
  rows.slice(0, shown).forEach(h => { const tr = document.createElement('tr');
    tr.innerHTML = '<td>'+h.date+'</td><td>'+h.session+'</td><td>'+h.position+'</td><td>'+cards(h.cards_list)+'</td><td>'
      +(h.board_list.length?cards(h.board_list):'–')+'</td><td class="'+(h.result>=0?'pos':'neg')+'">'+fmt(h.result)+'</td><td>'
      +(h.hors_fiche?'<span class="ko">oui</span>':'non')+'</td>';
    tr.addEventListener('click', () => { location.hash = h.id; }); body.appendChild(tr); });
  $('#count').textContent = rows.length + ' main(s)' + (rows.length>shown ? ' — '+shown+' affichées' : '');
  $('#more').style.display = rows.length>shown ? '' : 'none';
}
let cur = null, k = 0;
function seatPos(i, n, heroIdx){ const rel = (i - heroIdx + n) % n; const a = Math.PI/2 + rel * 2*Math.PI/n;
  return [50 + 44*Math.cos(a), 50 + 40*Math.sin(a)]; }
function renderReplay(){
  if(!cur) return; const r = cur; const st = r.steps[k]; const n = r.players.length;
  const heroIdx = r.players.findIndex(p => p.hero); const last = {}; const folded = new Set();
  for(let j=0;j<=k;j++){ const s = r.steps[j]; if(s.seat===null||s.seat===undefined) continue;
    if(s.street===st.street) last[s.seat] = s.text; if(s.text.startsWith('se couche')) folded.add(s.seat); }
  const end = k === r.steps.length-1;
  let html = '<div class="center"><div>'+(st.board.length?cards(st.board):'<span class="legend-note" style="color:#f0efec">'+STREETS[st.street]+'</span>')+'</div>'
    + '<div class="pot">pot '+st.pot.toFixed(1).replace('.',',')+' bb</div></div>';
  r.players.forEach((p,i) => { const [x,y] = seatPos(i,n,heroIdx);
    const showCards = p.hero || (end && p.cards.length && !folded.has(i));
    html += '<div class="seat'+(p.hero?' hero':'')+(st.seat===i?' act':'')+(folded.has(i)?' out':'')+'" style="left:'+x+'%;top:'+y+'%">'
      + '<div class="nm">'+p.position+' · '+p.name+'</div><div>'+p.stack_bb.toFixed(0)+' bb</div>'
      + (showCards ? cards(p.cards) : (folded.has(i)?'':cards(['??','??'], true)))
      + (last[i] ? '<span class="bub">'+last[i]+'</span>' : '') + '</div>'; });
  $('#felt').innerHTML = html;
  $('#step').textContent = (k+1)+' / '+r.steps.length+' — '+STREETS[st.street];
  const ol = $('#steps'); ol.innerHTML = '';
  r.steps.forEach((s,j) => { const li = document.createElement('li'); if(j===k) li.className='cur';
    const who = (s.seat===null||s.seat===undefined) ? STREETS[s.street] : r.players[s.seat].position+' '+r.players[s.seat].name;
    let t = who+' : '+s.text; if(s.fiche) t += ' — fiche : '+s.fiche+' '+(s.respect?'<span class="ok">✓</span>':'<span class="ko">✗ écart</span>');
    li.innerHTML = t; li.addEventListener('click', () => { k = j; renderReplay(); }); ol.appendChild(li); });
}
function openHand(id){ const r = REPLAYS[id]; const box = $('#replay');
  if(!r){ box.style.display='none'; return; } cur = r; k = 0; box.style.display = '';
  $('#rp-title').textContent = 'Main '+r.id+(r.date?' — '+r.date:'')+' — résultat '+fmt(r.result_bb)+' bb';
  renderReplay(); box.scrollIntoView({behavior:'smooth', block:'start'}); }
document.addEventListener('DOMContentLoaded', () => {
  ['#q','#f-session','#f-pos','#f-res','#f-hf'].forEach(s => $(s).addEventListener('input', () => { shown = 200; renderList(); }));
  $('#more').addEventListener('click', () => { shown += 200; renderList(); });
  $('#prev').addEventListener('click', () => { if(cur && k>0){ k--; renderReplay(); } });
  $('#next').addEventListener('click', () => { if(cur && k<cur.steps.length-1){ k++; renderReplay(); } });
  document.addEventListener('keydown', e => { if(!cur) return; if(e.key==='ArrowLeft'&&k>0){k--;renderReplay();} if(e.key==='ArrowRight'&&k<cur.steps.length-1){k++;renderReplay();} });
  window.addEventListener('hashchange', () => openHand(decodeURIComponent(location.hash.slice(1))));
  renderList(); if(location.hash) openHand(decodeURIComponent(location.hash.slice(1)));
});
})();
"""


def site_js() -> str:
    dark = json.dumps({k.lower(): v for k, v in theme.DARK_MAP.items()})
    return SITE_JS % {"dark": dark}
