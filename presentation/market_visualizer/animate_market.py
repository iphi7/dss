from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Create a self-contained HTML market animation.')
    ap.add_argument('--result-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--max-days', type=int, default=0, help='0 = all logged days')
    ap.add_argument('--max-orders-per-day', type=int, default=80)
    return ap.parse_args()


def firm_layout(firms: pd.DataFrame, edges: pd.DataFrame) -> dict[int, tuple[float, float]]:
    firm_ids = firms['firm_id'].astype(int).tolist()
    sectors = firms.set_index('firm_id')['sector'].to_dict()
    deg = {i: 0 for i in firm_ids}
    if not edges.empty:
        for _, r in edges.iterrows():
            deg[int(r['src_firm_id'])] = deg.get(int(r['src_firm_id']), 0) + 1
            deg[int(r['dst_firm_id'])] = deg.get(int(r['dst_firm_id']), 0) + 1
    by_sec: dict[int, list[int]] = {}
    for fid in firm_ids:
        by_sec.setdefault(int(sectors[fid]), []).append(fid)
    sec_ids = sorted(by_sec)
    centers = {}
    for k, sec in enumerate(sec_ids):
        th = 2 * math.pi * k / max(len(sec_ids), 1)
        centers[sec] = (0.5 + 0.30 * math.cos(th), 0.46 + 0.30 * math.sin(th))
    pos = {}
    for sec, ids in by_sec.items():
        cx, cy = centers[sec]
        ids = sorted(ids, key=lambda f: -deg.get(f, 0))
        for k, fid in enumerate(ids):
            th = 2 * math.pi * k / max(len(ids), 1) + 0.7
            r = 0.030 + 0.016 * math.sqrt(k)
            pos[fid] = (cx + r * math.cos(th), cy + r * math.sin(th))
    return pos


def to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    df = df.replace({np.nan: None})
    return df.to_dict(orient='records')


def build_payload(result_dir: Path, max_days: int, max_orders_per_day: int) -> dict:
    paths = pd.read_csv(result_dir / 'generated_paths.csv')
    firms = pd.read_csv(result_dir / 'firms.csv')
    firm_states = pd.read_csv(result_dir / 'firm_states.csv')
    investor_states = pd.read_csv(result_dir / 'investor_states.csv')
    orders = pd.read_csv(result_dir / 'orders.csv')
    edges = pd.read_csv(result_dir / 'true_graph_edges.csv')

    days = sorted(firm_states['day'].dropna().astype(int).unique().tolist())
    if max_days > 0:
        days = days[:max_days]
        firm_states = firm_states[firm_states['day'].isin(days)]
        investor_states = investor_states[investor_states['day'].isin(days)]
        orders = orders[orders['day'].isin(days)]
        paths = paths.iloc[:max(days) + 1]

    if not orders.empty and max_orders_per_day > 0:
        orders = (orders.sort_values(['day', 'value'], ascending=[True, False])
                        .groupby('day', as_index=False, group_keys=False)
                        .head(max_orders_per_day))

    # 日次の市場全体状態 (firm_states の各日1行目から抽出)
    meta_cols = [c for c in ['market_vol_t', 'common_noise', 'jump_abs',
                             'down_var_ewma', 'disaster_intensity'] if c in firm_states.columns]
    day_meta = (firm_states.groupby('day', as_index=False).first()[['day', 'Date'] + meta_cols])

    pos = firm_layout(firms, edges)
    firms_meta = firms.copy()
    firms_meta['x'] = firms_meta['firm_id'].astype(int).map(lambda i: pos[int(i)][0])
    firms_meta['y'] = firms_meta['firm_id'].astype(int).map(lambda i: pos[int(i)][1])

    paths = paths.reset_index().rename(columns={'index': 'day'})
    return {
        'days': days,
        'paths': to_records(paths),
        'firms': to_records(firms_meta),
        'firm_states': to_records(firm_states),
        'investor_states': to_records(investor_states),
        'orders': to_records(orders),
        'edges': to_records(edges),
        'day_meta': to_records(day_meta),
    }


HTML_TMPL = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<title>market_visualizer</title>
<style>
  :root { --bg:#0b1020; --panel:#111827; --line:#243244; --text:#e5e7eb; --dim:#94a3b8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Noto Sans CJK JP", sans-serif;
         background: var(--panel); color: var(--text); }
  .wrap { display:grid; grid-template-rows: 52px 1fr; height:100vh; }
  header { display:flex; align-items:center; gap:10px; padding:0 14px;
           border-bottom:1px solid var(--line); background:#0d1426; }
  header .date { font-size:17px; font-weight:600; min-width:210px; }
  header .date span { color:var(--dim); font-weight:400; font-size:13px; margin-left:8px; }
  .btn { background:#1e293b; color:var(--text); border:1px solid #334155; border-radius:6px;
         padding:5px 12px; cursor:pointer; font-size:13px; }
  .btn:hover { background:#27374e; }
  .btn.primary { background:#0ea5e9; border-color:#0ea5e9; color:#06202e; font-weight:600; min-width:74px; }
  select { background:#1e293b; color:var(--text); border:1px solid #334155; border-radius:6px; padding:4px 6px; }
  input[type=range] { flex:1; accent-color:#0ea5e9; }
  .crisis-badge { display:none; background:#7f1d1d; color:#fecaca; border:1px solid #ef4444;
                  border-radius:6px; padding:3px 10px; font-size:12px; font-weight:600; }
  main { display:grid; grid-template-columns: 1fr 400px; min-height:0; }
  #cv { width:100%; height:100%; display:block; background:var(--bg); }
  aside { border-left:1px solid var(--line); padding:12px 14px; overflow:auto; font-size:13px; }
  .chart { margin-bottom:10px; }
  .chart .t { color:var(--dim); font-size:11.5px; margin-bottom:2px; display:flex; justify-content:space-between; }
  .chart canvas { width:100%; height:74px; background:var(--bg); border-radius:6px; display:block; }
  table.orders { width:100%; border-collapse:collapse; font-size:12px; margin-top:4px; }
  table.orders td { padding:2px 4px; border-bottom:1px solid var(--line); }
  .buy { color:#22c55e; } .sell { color:#ef4444; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 12px; margin:8px 0; font-size:12px; color:var(--dim); }
  .legend .sw { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; }
  #tip { position:fixed; pointer-events:none; background:rgba(13,20,38,0.95); color:var(--text);
         border:1px solid #334155; border-radius:6px; padding:7px 10px; font-size:12px; display:none;
         z-index:10; max-width:260px; line-height:1.5; }
  h3 { margin:12px 0 4px; font-size:13px; color:var(--dim); text-transform:uppercase; letter-spacing:0.06em; }
  .kv { display:grid; grid-template-columns:auto auto; gap:1px 12px; font-size:12.5px; }
  .kv div:nth-child(odd) { color:var(--dim); }
  .hint { color:#64748b; font-size:11.5px; margin-top:10px; line-height:1.6; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="date"><span id="dayLabel">day 0</span><br><b id="dateLabel"></b></div>
    <button class="btn primary" id="playBtn">▶ Play</button>
    <button class="btn" id="back10">−10</button>
    <button class="btn" id="back1">−1</button>
    <button class="btn" id="fwd1">+1</button>
    <button class="btn" id="fwd10">+10</button>
    <select id="speed">
      <option value="600">0.5×</option>
      <option value="300" selected>1×</option>
      <option value="140">2×</option>
      <option value="60">4×</option>
    </select>
    <input id="slider" type="range" min="0" max="0" value="0">
    <div class="crisis-badge" id="crisisBadge">⚠ CRISIS</div>
  </header>
  <main>
    <canvas id="cv"></canvas>
    <aside>
      <div class="chart"><div class="t"><span>SP500 水準</span><span id="spVal"></span></div><canvas id="chSp"></canvas></div>
      <div class="chart"><div class="t"><span>DGS10 水準 (%)</span><span id="dgVal"></span></div><canvas id="chDg"></canvas></div>
      <div class="chart"><div class="t"><span>実現ボラ (market_vol_t)</span><span id="volVal"></span></div><canvas id="chVol"></canvas></div>
      <div class="chart"><div class="t"><span>富の集中度 HHI / 平均|不均衡|</span><span id="hhiVal"></span></div><canvas id="chHhi"></canvas></div>
      <h3>当日の市場</h3>
      <div class="kv" id="kv"></div>
      <h3>セクター</h3>
      <div class="legend" id="secLegend"></div>
      <div class="legend">
        <span><span class="sw" style="background:#22c55e"></span>上昇 / 買い注文</span>
        <span><span class="sw" style="background:#ef4444"></span>下落 / 売り注文</span>
        <span><span class="sw" style="background:#38bdf8"></span>投資家 (大きさ=富)</span>
      </div>
      <h3>Top orders</h3>
      <table class="orders"><tbody id="ordersBody"></tbody></table>
      <div class="hint">
        操作: Space=再生/停止, ←/→=±1日, Shift+←/→=±10日。<br>
        企業ノードにマウスで詳細、クリックで固定ハイライト (エッジと注文)。<br>
        ノード: 塗り=当日リターン、大きさ=市場ウェイト、縁=セクター色。危機中は画面が赤く縁取られます。
      </div>
    </aside>
  </main>
</div>
<div id="tip"></div>
<script>
const DATA = __PAYLOAD__;
const SEC_COLORS = ['#60a5fa','#f472b6','#fbbf24','#34d399','#c084fc','#fb923c','#2dd4bf','#a3e635',
                    '#f87171','#818cf8','#facc15','#4ade80'];
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const tip = document.getElementById('tip');
const slider = document.getElementById('slider');
let idx = 0, timer = null, hoverFirm = null, pinnedFirm = null, hoverInv = null;
slider.max = Math.max(DATA.days.length - 1, 0);

const firms = new Map(DATA.firms.map(d => [d.firm_id, d]));
const groupBy = (arr, key) => { const m = new Map(); for (const r of arr){ if(!m.has(r[key])) m.set(r[key], []); m.get(r[key]).push(r); } return m; };
const firmByDay = groupBy(DATA.firm_states, 'day');
const invByDay = groupBy(DATA.investor_states, 'day');
const ordByDay = groupBy(DATA.orders, 'day');
const pathByDay = new Map(DATA.paths.map(r => [r.day, r]));
const metaByDay = new Map((DATA.day_meta||[]).map(r => [r.day, r]));
const neighbors = new Map();
for (const e of DATA.edges){
  if(!neighbors.has(e.src_firm_id)) neighbors.set(e.src_firm_id, new Set());
  if(!neighbors.has(e.dst_firm_id)) neighbors.set(e.dst_firm_id, new Set());
  neighbors.get(e.src_firm_id).add(e.dst_firm_id);
  neighbors.get(e.dst_firm_id).add(e.src_firm_id);
}
const sectors = [...new Set(DATA.firms.map(d => d.sector))].sort((a,b)=>a-b);
const secColor = s => SEC_COLORS[sectors.indexOf(s) % SEC_COLORS.length];
document.getElementById('secLegend').innerHTML =
  sectors.map(s => `<span><span class="sw" style="background:${secColor(s)}"></span>sector ${s}</span>`).join('');

// レイアウト座標 → 画面座標 (ネットワークは左 72% 領域に描画)
function sx(x){ return x * cv.width; }
function sy(y){ return y * cv.height * 0.82; }
function investorPos(k, n){ const x = 0.06 + 0.88 * (k + 0.5) / Math.max(n, 1); return [x * cv.width, cv.height * 0.93]; }
function colorRet(r){
  const v = Math.max(-0.06, Math.min(0.06, r || 0));
  const t = Math.abs(v) / 0.06;
  if (v >= 0) return `rgba(${34+30*(1-t)},${120+110*t},94,${0.55+0.45*t})`;
  return `rgba(${150+90*t},${70-30*t},70,${0.55+0.45*t})`;
}

// ---- ヒットテスト用キャッシュ ----
let firmHit = [], invHit = [];

function draw(){
  if(DATA.days.length === 0) return;
  const day = DATA.days[idx];
  const W = cv.width, H = cv.height, dpr = devicePixelRatio;
  ctx.clearRect(0,0,W,H); ctx.fillStyle = '#0b1020'; ctx.fillRect(0,0,W,H);
  const meta = metaByDay.get(day) || {};
  const crisis = (meta.disaster_intensity || 0) > 0.01;

  const focus = pinnedFirm ?? hoverFirm;
  // エッジ
  for(const e of DATA.edges){
    const a = firms.get(e.src_firm_id), b = firms.get(e.dst_firm_id); if(!a || !b) continue;
    const hot = focus !== null && (e.src_firm_id === focus || e.dst_firm_id === focus);
    ctx.strokeStyle = hot ? 'rgba(125,211,252,0.65)' : 'rgba(148,163,184,0.10)';
    ctx.lineWidth = (hot ? 1.6 : 0.7) * dpr;
    ctx.beginPath(); ctx.moveTo(sx(a.x), sy(a.y)); ctx.lineTo(sx(b.x), sy(b.y)); ctx.stroke();
  }
  // 投資家
  const invs = invByDay.get(day) || [];
  const invIndex = new Map(invs.map((d,k) => [d.investor_id, k]));
  invHit = [];
  for(let k=0;k<invs.length;k++){
    const d = invs[k]; const [x,y] = investorPos(k, invs.length);
    const rad = (3 + 4.5*Math.log10(1 + Math.max(d.wealth, 0))) * dpr;
    const hot = hoverInv === d.investor_id;
    ctx.fillStyle = hot ? '#7dd3fc' : '#38bdf8'; ctx.globalAlpha = hot ? 1 : 0.78;
    ctx.beginPath(); ctx.arc(x,y,rad,0,2*Math.PI); ctx.fill(); ctx.globalAlpha = 1;
    invHit.push({x, y, r: rad + 3*dpr, d});
  }
  // 注文 (曲線 + 企業側で終端)
  const orders = ordByDay.get(day) || [];
  for(const o of orders){
    const f = firms.get(o.firm_id); const k = invIndex.get(o.investor_id);
    if(!f || k === undefined) continue;
    const related = focus !== null && o.firm_id === focus || (hoverInv !== null && o.investor_id === hoverInv);
    if (focus !== null || hoverInv !== null){ if(!related){ continue; } }
    const [ix,iy] = investorPos(k, invs.length); const fx=sx(f.x), fy=sy(f.y);
    const alpha = Math.min(0.8, 0.15 + Math.log1p(o.value) / 11);
    ctx.strokeStyle = o.side === 'buy' ? `rgba(34,197,94,${alpha})` : `rgba(239,68,68,${alpha})`;
    ctx.lineWidth = Math.max(0.6, Math.min(4, Math.log1p(o.value))) * dpr;
    const mx = (ix+fx)/2 + (fy-iy)*0.08, my = (iy+fy)/2 - (fx-ix)*0.08;
    ctx.beginPath(); ctx.moveTo(ix,iy); ctx.quadraticCurveTo(mx,my,fx,fy); ctx.stroke();
  }
  // 企業ノード
  const fs = firmByDay.get(day) || [];
  firmHit = [];
  const byWeight = [...fs].sort((a,b)=>(b.market_weight||0)-(a.market_weight||0));
  const labelSet = new Set(byWeight.slice(0,5).map(d=>d.firm_id));
  for(const d of fs){
    const f = firms.get(d.firm_id); if(!f) continue;
    const x = sx(f.x), y = sy(f.y);
    const rad = (4 + 46*Math.sqrt(Math.max(d.market_weight || 0, 0))) * dpr;
    ctx.fillStyle = colorRet(d.return);
    ctx.beginPath(); ctx.arc(x, y, rad, 0, 2*Math.PI); ctx.fill();
    ctx.strokeStyle = secColor(f.sector); ctx.lineWidth = (focus === d.firm_id ? 2.4 : 1.1) * dpr; ctx.stroke();
    if (labelSet.has(d.firm_id)){
      ctx.fillStyle = 'rgba(226,232,240,0.85)'; ctx.font = `${10.5*dpr}px sans-serif`;
      ctx.fillText(`#${d.firm_id}`, x + rad + 2*dpr, y + 3*dpr);
    }
    firmHit.push({x, y, r: rad + 3*dpr, d, f});
  }
  // 危機ヴィネット
  if (crisis){
    const g = ctx.createRadialGradient(W/2, H/2, Math.min(W,H)*0.42, W/2, H/2, Math.max(W,H)*0.72);
    g.addColorStop(0, 'rgba(127,29,29,0)'); g.addColorStop(1, `rgba(185,28,28,${Math.min(0.5, 0.25+0.3*meta.disaster_intensity)})`);
    ctx.fillStyle = g; ctx.fillRect(0,0,W,H);
  }
  document.getElementById('crisisBadge').style.display = crisis ? 'block' : 'none';
  drawSidePanels(day, meta, orders);
}

// ---- 右パネルのチャート ----
function drawLineChart(canvas, series, key, color, day, opts={}){
  const c = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * devicePixelRatio;
  c.clearRect(0,0,W,H);
  const vals = series.map(d => +d[key]).filter(Number.isFinite);
  if (vals.length < 2) return;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-12){ hi = lo + 1; }
  const px = i => W * i / Math.max(series.length - 1, 1);
  const py = v => H * (0.94 - 0.86 * (v - lo) / (hi - lo));
  // 危機網掛け
  if (opts.crisisKey){
    c.fillStyle = 'rgba(239,68,68,0.13)';
    series.forEach((d, i) => { if ((metaByDay.get(d.day)?.disaster_intensity || 0) > 0.01)
      c.fillRect(px(i) - W/(2*series.length), 0, W/series.length + 1, H); });
  }
  c.strokeStyle = 'rgba(148,163,184,0.25)'; c.lineWidth = 1; c.setLineDash([3,4]);
  [lo, hi].forEach(v => { c.beginPath(); c.moveTo(0, py(v)); c.lineTo(W, py(v)); c.stroke(); });
  c.setLineDash([]);
  c.strokeStyle = color; c.lineWidth = 1.6 * devicePixelRatio; c.beginPath();
  series.forEach((d, i) => { const y = py(+d[key]); if(i===0) c.moveTo(px(i), y); else c.lineTo(px(i), y); });
  c.stroke();
  if (opts.key2){
    c.strokeStyle = opts.color2; c.lineWidth = 1.2 * devicePixelRatio; c.beginPath();
    const v2 = series.map(d => +d[opts.key2]).filter(Number.isFinite);
    const lo2 = Math.min(...v2), hi2 = Math.max(...v2, lo2 + 1e-12);
    series.forEach((d, i) => { const y = H * (0.94 - 0.86 * ((+d[opts.key2]) - lo2) / (hi2 - lo2));
      if(i===0) c.moveTo(px(i), y); else c.lineTo(px(i), y); });
    c.stroke();
  }
  // 現在日マーカー
  const cur = series.findIndex(d => d.day === day);
  if (cur >= 0){
    c.strokeStyle = 'rgba(226,232,240,0.8)'; c.lineWidth = 1 * devicePixelRatio;
    c.beginPath(); c.moveTo(px(cur), 0); c.lineTo(px(cur), H); c.stroke();
    c.fillStyle = color; c.beginPath(); c.arc(px(cur), py(+series[cur][key]), 3*devicePixelRatio, 0, 2*Math.PI); c.fill();
  }
  // min/max ラベル
  c.fillStyle = 'rgba(148,163,184,0.85)'; c.font = `${9.5*devicePixelRatio}px sans-serif`;
  c.fillText(hi.toPrecision(4), 4*devicePixelRatio, py(hi) - 3*devicePixelRatio);
  c.fillText(lo.toPrecision(4), 4*devicePixelRatio, py(lo) + 10*devicePixelRatio);
}

function fmt(v, d=3){ return (v===null||v===undefined||!Number.isFinite(+v)) ? '–' : (+v).toFixed(d); }

function drawSidePanels(day, meta, orders){
  const upto = DATA.paths.filter(d => d.day <= day);
  const metaUpto = (DATA.day_meta||[]).filter(d => d.day <= day);
  const p = pathByDay.get(day) || {};
  drawLineChart(document.getElementById('chSp'), upto, 'sp500_abs', '#f8fafc', day, {crisisKey:1});
  drawLineChart(document.getElementById('chDg'), upto, 'DGS10_abs', '#f59e0b', day, {crisisKey:1});
  drawLineChart(document.getElementById('chVol'), metaUpto, 'market_vol_t', '#f472b6', day, {});
  drawLineChart(document.getElementById('chHhi'), upto, '_hhi', '#34d399', day, {key2:'_imbal', color2:'rgba(96,165,250,0.8)'});
  document.getElementById('spVal').textContent = fmt(p.sp500_abs, 2) + '  (r ' + fmt((p.sp500||0)*100, 2) + '%)';
  document.getElementById('dgVal').textContent = fmt(p.DGS10_abs, 3) + '  (Δ ' + fmt(p.DGS10, 3) + ')';
  document.getElementById('volVal').textContent = fmt(meta.market_vol_t, 4);
  document.getElementById('hhiVal').textContent = fmt(p._hhi, 3) + ' / ' + fmt(p._imbal, 3);
  document.getElementById('dayLabel').textContent = `day ${day} / ${DATA.days[DATA.days.length-1]}`;
  document.getElementById('dateLabel').textContent = p.Date || '';
  const kv = {
    '共通ノイズ': fmt(meta.common_noise, 4), 'ジャンプ量': fmt(meta.jump_abs, 4),
    '危機強度': fmt(meta.disaster_intensity, 3), '下方分散EWMA': meta.down_var_ewma ? (+meta.down_var_ewma).toExponential(2) : '–',
    '表示注文数': orders.length,
  };
  document.getElementById('kv').innerHTML = Object.entries(kv).map(([k,v]) => `<div>${k}</div><div>${v}</div>`).join('');
  document.getElementById('ordersBody').innerHTML = orders.slice(0,10).map(o =>
    `<tr><td class="${o.side}">${o.side}</td><td>inv${o.investor_id}</td><td>→ firm${o.firm_id}</td>` +
    `<td style="text-align:right">${Number(o.value).toExponential(2)}</td></tr>`).join('');
}

// ---- インタラクション ----
cv.addEventListener('mousemove', ev => {
  const rect = cv.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * devicePixelRatio, y = (ev.clientY - rect.top) * devicePixelRatio;
  let hitF = null, hitI = null;
  for (const h of firmHit){ if ((x-h.x)**2 + (y-h.y)**2 <= h.r**2){ hitF = h; break; } }
  if (!hitF) for (const h of invHit){ if ((x-h.x)**2 + (y-h.y)**2 <= h.r**2){ hitI = h; break; } }
  const newHoverF = hitF ? hitF.d.firm_id : null, newHoverI = hitI ? hitI.d.investor_id : null;
  if (newHoverF !== hoverFirm || newHoverI !== hoverInv){ hoverFirm = newHoverF; hoverInv = newHoverI; draw(); }
  if (hitF){
    const d = hitF.d;
    tip.style.display = 'block'; tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 10) + 'px';
    tip.innerHTML = `<b>firm ${d.firm_id}</b> (sector ${hitF.f.sector}, 次数 ${(neighbors.get(d.firm_id)||new Set()).size})<br>` +
      `価格 ${fmt(d.price,3)} / リターン ${fmt((d.return||0)*100,2)}%<br>` +
      `市場ウェイト ${fmt((d.market_weight||0)*100,2)}% / 不均衡 ${fmt(d.imbalance,2)}<br>` +
      `買い ${Number(d.buy_value||0).toExponential(2)} / 売り ${Number(d.sell_value||0).toExponential(2)}`;
  } else if (hitI){
    const d = hitI.d;
    tip.style.display = 'block'; tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 10) + 'px';
    tip.innerHTML = `<b>investor ${d.investor_id}</b> (専門 sector ${d.expertise_sector})<br>` +
      `富 ${Number(d.wealth||0).toExponential(2)} (現金 ${Number(d.cash||0).toExponential(2)})<br>` +
      `当日 買い ${Number(d.buy_value||0).toExponential(2)} / 売り ${Number(d.sell_value||0).toExponential(2)}<br>` +
      `ボラ選好 ${fmt(d.vol_sensitivity,2)} / グラフ精度 ${fmt(d.graph_quality,2)}`;
  } else { tip.style.display = 'none'; }
});
cv.addEventListener('mouseleave', () => { tip.style.display='none'; if(hoverFirm!==null||hoverInv!==null){hoverFirm=null;hoverInv=null;draw();} });
cv.addEventListener('click', () => { pinnedFirm = (hoverFirm !== null && hoverFirm !== pinnedFirm) ? hoverFirm : null; draw(); });

// ---- 再生制御 ----
function setIdx(v){ idx = Math.max(0, Math.min(DATA.days.length-1, v)); slider.value = idx; draw(); }
function playing(){ return timer !== null; }
function stop(){ clearInterval(timer); timer = null; document.getElementById('playBtn').textContent = '▶ Play'; }
function start(){ if (playing()) return;
  const iv = +document.getElementById('speed').value;
  timer = setInterval(() => { if (idx >= DATA.days.length-1) stop(); else setIdx(idx+1); }, iv);
  document.getElementById('playBtn').textContent = '❚❚ Pause'; }
document.getElementById('playBtn').onclick = () => playing() ? stop() : start();
document.getElementById('speed').onchange = () => { if (playing()){ stop(); start(); } };
document.getElementById('back1').onclick = () => setIdx(idx-1);
document.getElementById('fwd1').onclick = () => setIdx(idx+1);
document.getElementById('back10').onclick = () => setIdx(idx-10);
document.getElementById('fwd10').onclick = () => setIdx(idx+10);
slider.addEventListener('input', e => setIdx(+e.target.value));
window.addEventListener('keydown', e => {
  if (e.code === 'Space'){ e.preventDefault(); playing() ? stop() : start(); }
  else if (e.key === 'ArrowRight'){ setIdx(idx + (e.shiftKey ? 10 : 1)); }
  else if (e.key === 'ArrowLeft'){ setIdx(idx - (e.shiftKey ? 10 : 1)); }
});
function resize(){ cv.width = cv.clientWidth * devicePixelRatio; cv.height = cv.clientHeight * devicePixelRatio; draw(); }
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    out = args.output or (args.result_dir / 'market_animation.html')
    payload = build_payload(args.result_dir, args.max_days, args.max_orders_per_day)
    html = HTML_TMPL.replace('__PAYLOAD__', json.dumps(payload, ensure_ascii=False))
    out.write_text(html, encoding='utf-8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
