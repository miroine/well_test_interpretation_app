"""Self-contained HTML/JS buildup animation for st.components.v1.html.

All numerics arrive as a JSON payload from engine.viz.animation_payload();
the JavaScript only draws. No external libraries (works offline / in iframes).
"""
from __future__ import annotations

import json

_TEMPLATE = r"""
<style>
  #wrap{font-family:Inter,Segoe UI,Arial,sans-serif;color:#051C2C}
  #wrap .bar{display:flex;gap:10px;align-items:center;margin:0 0 8px 0;flex-wrap:wrap}
  #wrap button,#wrap select{font:inherit;padding:5px 10px;border:1px solid #9bb0c8;
      background:#fff;color:#051C2C;border-radius:6px;cursor:pointer}
  #wrap button:hover{background:#eef3fb}
  #wrap input[type=range]{flex:1;min-width:140px;accent-color:#2251FF}
  #wrap .read{font-size:13px;font-variant-numeric:tabular-nums;color:#33475b}
  #wrap .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  #wrap .ttl{font-size:12px;font-weight:600;letter-spacing:.02em;margin:2px 0 3px 2px;color:#33475b}
  #wrap canvas{width:100%;height:290px;background:#fff;border:1px solid #dbe3ee;border-radius:6px;display:block}
  #wrap .foot{font-size:11.5px;color:#5b6f85;margin-top:8px;line-height:1.4}
  #wrap .badge{display:inline-block;padding:2px 8px;border-radius:10px;background:#e8efff;
      color:#1b3fc9;font-weight:600;font-size:12px}
  @media (max-width:760px){#wrap .grid{grid-template-columns:1fr}}
</style>
<div id="wrap">
  <div class="bar">
    <button id="play">Pause</button>
    <button id="rst">Restart</button>
    <label class="read">Speed
      <select id="spd"><option value="0.5">0.5x</option><option value="1" selected>1x</option>
      <option value="2">2x</option><option value="4">4x</option></select></label>
    <input type="range" id="scrub" min="0" value="0" step="1">
    <span class="read" id="read"></span>
  </div>
  <div class="grid">
    <div><div class="ttl">Reservoir pressure map (plan view)</div><canvas id="c1"></canvas></div>
    <div><div class="ttl">Pressure vs radius</div><canvas id="c2"></canvas></div>
    <div><div class="ttl">Shut-in pressure vs time</div><canvas id="c3"></canvas></div>
    <div><div class="ttl">Diagnostic plot: pressure change and derivative</div><canvas id="c4"></canvas></div>
  </div>
  <div class="foot" id="foot"></div>
</div>
<script>
(function(){
const D = __DATA__;
const nF = D.prof.length;
const STOPS = [[0,[11,42,74]],[0.35,[34,81,255]],[0.7,[127,209,255]],[1,[234,246,255]]];
let f = 0, playing = true, speed = 1, lastT = 0, endHold = 0;
const $ = id => document.getElementById(id);
const scrub = $('scrub'); scrub.max = nF - 1;

let pmin = Infinity;
for (const row of D.prof) for (const v of row) if (v < pmin) pmin = v;
const pmax = D.pi;
const pwsMin = Math.min.apply(null, D.pws), pwsMax = Math.max.apply(null, D.pws);

function colorAt(x){
  x = Math.max(0, Math.min(1, x));
  for (let i = 1; i < STOPS.length; i++){
    if (x <= STOPS[i][0]){
      const a = STOPS[i-1], b = STOPS[i], t = (x - a[0]) / (b[0] - a[0]);
      return 'rgb(' + [0,1,2].map(k => Math.round(a[1][k] + (b[1][k] - a[1][k]) * t)).join(',') + ')';
    }
  }
  return 'rgb(234,246,255)';
}
function interpLogR(prof, rft){
  const R = D.r, n = R.length;
  if (rft <= R[0]) return prof[0];
  if (rft >= R[n-1]) return prof[n-1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1){ const m = (lo + hi) >> 1; if (R[m] <= rft) lo = m; else hi = m; }
  const t = (Math.log(rft) - Math.log(R[lo])) / (Math.log(R[hi]) - Math.log(R[lo]));
  return prof[lo] + (prof[hi] - prof[lo]) * t;
}
function fit(c){
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w*dpr) || c.height !== Math.round(h*dpr)){ c.width = Math.round(w*dpr); c.height = Math.round(h*dpr); }
  const g = c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
  return {g:g, w:w, h:h};
}
function niceStep(range, n){
  const raw = range / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
  return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
}
function fmt(v){
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-2)) return v.toExponential(0).replace('e+','e');
  return (Math.round(v*1000)/1000).toString();
}
// generic axes. xr,yr are [min,max] in plotted units (log10 if logx/logy)
function axes(g, W, H, xr, yr, logx, logy, xl, yl){
  const m = {l:50, r:12, t:10, b:34};
  const px = v => m.l + (v - xr[0]) / (xr[1] - xr[0]) * (W - m.l - m.r);
  const py = v => H - m.b - (v - yr[0]) / (yr[1] - yr[0]) * (H - m.t - m.b);
  g.strokeStyle = '#e3e9f2'; g.fillStyle = '#5b6f85'; g.lineWidth = 1; g.font = '10.5px Inter,Arial,sans-serif';
  function ticks(r, lg){
    const out = [];
    if (lg){ for (let e = Math.ceil(r[0]); e <= Math.floor(r[1]); e++) out.push([e, fmt(Math.pow(10, e))]); }
    else { const s = niceStep(r[1]-r[0], 5); for (let v = Math.ceil(r[0]/s)*s; v <= r[1]+1e-9; v += s) out.push([v, fmt(v)]); }
    return out;
  }
  g.textAlign = 'center';
  for (const t of ticks(xr, logx)){ const x = px(t[0]); g.beginPath(); g.moveTo(x, m.t); g.lineTo(x, H-m.b); g.stroke(); g.fillText(t[1], x, H-m.b+13); }
  g.textAlign = 'right';
  for (const t of ticks(yr, logy)){ const y = py(t[0]); g.beginPath(); g.moveTo(m.l, y); g.lineTo(W-m.r, y); g.stroke(); g.fillText(t[1], m.l-5, y+3); }
  g.strokeStyle = '#9bb0c8'; g.strokeRect(m.l, m.t, W-m.l-m.r, H-m.t-m.b);
  g.fillStyle = '#33475b'; g.textAlign = 'center'; g.font = '11px Inter,Arial,sans-serif';
  g.fillText(xl, m.l + (W-m.l-m.r)/2, H-4);
  g.save(); g.translate(11, m.t + (H-m.t-m.b)/2); g.rotate(-Math.PI/2); g.fillText(yl, 0, 0); g.restore();
  return {px:px, py:py, m:m};
}
function line(g, pts, color, width, dash){
  g.save(); g.strokeStyle = color; g.lineWidth = width; if (dash) g.setLineDash(dash);
  g.beginPath(); let open = false;
  for (const p of pts){ if (p === null){ open = false; continue; } if (!open){ g.moveTo(p[0], p[1]); open = true; } else g.lineTo(p[0], p[1]); }
  g.stroke(); g.restore();
}

function drawMap(k){
  const c = fit($('c1')), g = c.g, W = c.w, H = c.h;
  const cx = W/2, cy = H/2, R = Math.min(W, H)/2 - 12, Rv = D.rmax;
  const prof = D.prof[k];
  g.save(); g.beginPath(); g.arc(cx, cy, R, 0, 2*Math.PI); g.clip();
  for (let rho = R; rho >= 0; rho -= 1){
    const rft = Math.max(rho / R * Rv, D.rw);
    g.beginPath(); g.arc(cx, cy, rho, 0, 2*Math.PI);
    g.fillStyle = colorAt((interpLogR(prof, rft) - pmin) / (pmax - pmin + 1e-9)); g.fill();
  }
  // boundaries: grey out the non-reservoir side and draw the line
  const b = D.boundary, sc = R / Rv;
  g.fillStyle = 'rgba(120,130,145,0.78)';
  const sealing = (b.type !== 'const_p_fault');
  if (b.type === 'fault' || b.type === 'const_p_fault'){
    g.fillRect(cx + b.d1*sc, cy - R, 2*R, 2*R);
  } else if (b.type === 'channel'){
    g.fillRect(cx - R, cy - R, 2*R, R - b.d1*sc); g.fillRect(cx - R, cy + b.d2*sc, 2*R, R);
  } else if (b.type === 'corner'){
    g.fillRect(cx + b.d1*sc, cy - R, 2*R, 2*R); g.fillRect(cx - R, cy - R, 2*R, R - b.d2*sc);
  }
  g.restore();
  g.lineWidth = 2.5; g.strokeStyle = sealing ? '#b3261e' : '#0a7d4f';
  function seg(x1,y1,x2,y2){ g.beginPath(); g.moveTo(x1,y1); g.lineTo(x2,y2); g.stroke(); }
  if (b.type === 'fault' || b.type === 'const_p_fault') seg(cx + b.d1*sc, cy - R, cx + b.d1*sc, cy + R);
  if (b.type === 'channel'){ seg(cx - R, cy - b.d1*sc, cx + R, cy - b.d1*sc); seg(cx - R, cy + b.d2*sc, cx + R, cy + b.d2*sc); }
  if (b.type === 'corner'){ seg(cx + b.d1*sc, cy - R, cx + b.d1*sc, cy + R); seg(cx - R, cy - b.d2*sc, cx + R, cy - b.d2*sc); }
  // radius of investigation
  const ri = D.ri[D.frame_idx[k]];
  if (ri * sc < R){ g.save(); g.setLineDash([5,4]); g.strokeStyle = '#ffffff'; g.lineWidth = 1.6;
    g.beginPath(); g.arc(cx, cy, ri*sc, 0, 2*Math.PI); g.stroke(); g.restore(); }
  g.strokeStyle = '#9bb0c8'; g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, R, 0, 2*Math.PI); g.stroke();
  g.fillStyle = '#051C2C'; g.beginPath(); g.arc(cx, cy, 3.5, 0, 2*Math.PI); g.fill();
  g.fillStyle = '#33475b'; g.font = '11px Inter,Arial,sans-serif'; g.textAlign = 'left';
  g.fillText('map radius ' + fmt(Rv) + ' (dashed ring = radius of investigation)', 8, H - 6);
}
function drawProfile(k){
  const c = fit($('c2')), g = c.g;
  const xr = [Math.log10(D.r[0]), Math.log10(D.r[D.r.length-1])];
  const pad = 0.04 * (pmax - pmin), yr = [pmin - pad, pmax + pad];
  const a = axes(g, c.w, c.h, xr, yr, true, false, 'radius from well', 'pressure');
  for (let j = Math.max(0, k-24); j < k; j += 6){
    line(g, D.r.map((r,i) => [a.px(Math.log10(r)), a.py(D.prof[j][i])]), 'rgba(34,81,255,0.18)', 1.2);
  }
  line(g, D.r.map((r,i) => [a.px(Math.log10(r)), a.py(D.prof[k][i])]), '#2251FF', 2.4);
  const ri = D.ri[D.frame_idx[k]];
  if (ri > D.r[0] && ri < D.r[D.r.length-1]){
    const x = a.px(Math.log10(ri)); line(g, [[x, a.m.t],[x, c.h - a.m.b]], '#b3261e', 1.3, [5,4]);
    g.fillStyle = '#b3261e'; g.font = '10.5px Inter,Arial,sans-serif'; g.textAlign = 'left'; g.fillText('r_inv', x + 4, a.m.t + 11);
  }
  line(g, [[a.m.l, a.py(pmax)],[c.w - a.m.r, a.py(pmax)]], '#0a7d4f', 1, [2,3]);
}
function drawPressure(k){
  const c = fit($('c3')), g = c.g, i = D.frame_idx[k];
  const lx = D.dt.map(Math.log10);
  const pad = 0.05 * (pwsMax - pwsMin), yr = [pwsMin - pad, pwsMax + pad];
  const a = axes(g, c.w, c.h, [lx[0], lx[lx.length-1]], yr, true, false, 'shut-in time, dt (hr)', 'pressure');
  const all = D.dt.map((d,j) => [a.px(lx[j]), a.py(D.pws[j])]);
  line(g, all, '#cdd7e5', 1.6);
  line(g, all.slice(0, i+1), '#2251FF', 2.6);
  g.fillStyle = '#2251FF'; g.beginPath(); g.arc(all[i][0], all[i][1], 4.5, 0, 2*Math.PI); g.fill();
}
function drawDiag(k){
  const c = fit($('c4')), g = c.g, i = D.frame_idx[k];
  const xs = [], ys = [];
  for (let j = 0; j < D.ta.length; j++){
    if (D.ta[j] > 0){ xs.push(Math.log10(D.ta[j])); }
    if (D.dp[j] > 0) ys.push(Math.log10(D.dp[j]));
    if (D.der[j] !== null && D.der[j] > 0) ys.push(Math.log10(D.der[j]));
  }
  const xr = [Math.min.apply(null, xs), Math.max.apply(null, xs)];
  const yr = [Math.min.apply(null, ys) - 0.15, Math.max.apply(null, ys) + 0.15];
  const a = axes(g, c.w, c.h, xr, yr, true, true, 'Agarwal equivalent time (hr)', 'dp and derivative (psi)');
  const mk = (arr, upto) => { const out = []; for (let j = 0; j <= upto; j++){ const v = arr[j]; out.push((v === null || v <= 0 || D.ta[j] <= 0) ? null : [a.px(Math.log10(D.ta[j])), a.py(Math.log10(v))]); } return out; };
  const n = D.ta.length - 1;
  line(g, mk(D.dp, n), '#d3dbe8', 1.5); line(g, mk(D.der, n), '#e1d4cf', 1.5);
  line(g, mk(D.dp, i), '#2251FF', 2.4); line(g, mk(D.der, i), '#c2410c', 2.4);
  for (const [arr, col] of [[D.dp, '#2251FF'], [D.der, '#c2410c']]){
    if (arr[i] !== null && arr[i] > 0){ g.fillStyle = col; g.beginPath(); g.arc(a.px(Math.log10(D.ta[i])), a.py(Math.log10(arr[i])), 4.5, 0, 2*Math.PI); g.fill(); }
  }
  g.font = '11px Inter,Arial,sans-serif'; g.textAlign = 'left';
  g.fillStyle = '#2251FF'; g.fillText('dp', a.m.l + 8, a.m.t + 14);
  g.fillStyle = '#c2410c'; g.fillText('derivative', a.m.l + 30, a.m.t + 14);
}
function readout(k){
  const i = D.frame_idx[k];
  $('read').innerHTML = 'dt = <b>' + fmt(D.dt[i]) + ' hr</b> &nbsp; p<sub>ws</sub> = <b>' + D.pws[i].toFixed(1) +
    '</b> &nbsp; r<sub>inv</sub> = <b>' + fmt(D.ri[i]) + '</b> &nbsp; <span class="badge">' + D.labels[i] + '</span>';
}
function draw(){
  drawMap(f); drawProfile(f); drawPressure(f); drawDiag(f); readout(f); scrub.value = f;
}
function setPlaying(v){ playing = v; $('play').textContent = v ? 'Pause' : 'Play'; }
$('play').onclick = function(){ if (!playing && f >= nF - 1) f = 0; setPlaying(!playing); };
$('rst').onclick = function(){ f = 0; setPlaying(true); draw(); };
$('spd').onchange = function(e){ speed = parseFloat(e.target.value); };
scrub.oninput = function(e){ f = parseInt(e.target.value, 10); setPlaying(false); draw(); };
window.addEventListener('resize', draw);
function loop(ts){
  if (playing && ts - lastT > 1000 / (14 * speed)){
    lastT = ts;
    if (f < nF - 1){ f++; draw(); } else { setPlaying(false); }
  }
  requestAnimationFrame(loop);
}
$('foot').innerHTML = '<b>' + D.title + '</b> &mdash; flow period ' + fmt(D.tp) + ' hr at ' + fmt(D.q) +
  ' (reservoir units/day), then shut-in. The map and radial profile show infinite-acting diffusion from the model sandface rate; ' +
  'boundaries are drawn for geometry (red = sealing, green = constant pressure) and act through the diagnostic curves.';
draw(); requestAnimationFrame(loop);
})();
</script>
"""


def build_html(payload: dict) -> str:
    """Return the animation page. Escapes '</' so JSON cannot close the script."""
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data)
