"""
ZA-FINAL6 で 60年×30パスを生成し、ランダムな5年窓 (1260日) を1000個切り出して
synthetic_returns_siggan_iter7_std_seed123_1000paths_wide.csv と同一の wide 形式
(step, date, sp500_path_i, dgs10_path_i) の1つの CSV にまとめる。

- sp500_path_i: 日次単純リターン、dgs10_path_i: 水準の日次差分 (%pt) — output.csv と同じ単位
- date 列は SigGAN ファイルと同一 (2026-07-08 起点の営業日 1260 行)
- 60年パスは histinit (1966年初期値)。窓の切り出しは専用rngで path×start を一様抽出
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ZA = Path(__file__).parent
sys.path.insert(0, str(ZA))
from model import Config                      # noqa: E402
from model_gpu import simulate_market_gpu     # noqa: E402
from za_final_config import ZA_FINAL6_PARAMS  # noqa: E402

N_LONG = 30          # 60年パス本数
N_OUT = 1000         # 切り出す5年窓の数
WIN = 1260           # 5年 = 1260営業日
SEED0 = 3001         # 60年パスの seed 起点
CUT_SEED = 424242    # 窓抽出用 rng

OUT = Path('/home/u00121/synthetic_returns_za_final6_1000paths_wide.csv')
TEMPLATE = '/home/u00121/synthetic_returns_siggan_iter7_std_seed123_1000paths_wide.csv'
LONG_DIR = ZA / 'results_fix6_2' / 'final6_60y_pool'
LONG_DIR.mkdir(parents=True, exist_ok=True)

hist = pd.read_csv('/home/u00121/output.csv')
n_days = len(hist) - 2
rt = hist.tail(n_days).reset_index(drop=True)
init_sp = float(rt.loc[0, 'sp500_abs'])
init_dg = float(rt.loc[0, 'DGS10_abs'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 60年×30パス (キャッシュ: 既存ファイルはスキップ) ----
long_paths = []
for i in range(N_LONG):
    seed = SEED0 + i
    f = LONG_DIR / f'long_seed{seed}.csv'
    if f.exists():
        g = pd.read_csv(f)
    else:
        print(f'generating 60y path {i+1}/{N_LONG} (seed {seed})', flush=True)
        c = Config(**{**ZA_FINAL6_PARAMS, 'seed': seed, 'n_days': n_days,
                      'initial_sp500_abs': init_sp, 'initial_dgs10_abs': init_dg})
        g, _, _, _ = simulate_market_gpu(hist, c, device=device)
        g = g[['sp500', 'DGS10']].astype(np.float32)
        g.to_csv(f, index=False)
    long_paths.append(g[['sp500', 'DGS10']].to_numpy(dtype=np.float32))
print('60y pool ready', flush=True)

# ---- ランダム5年窓の切り出し ----
cut_rng = np.random.default_rng(CUT_SEED)
tmpl = pd.read_csv(TEMPLATE, usecols=['step', 'date'])
assert len(tmpl) == WIN
cols = {'step': tmpl['step'], 'date': tmpl['date']}
meta_rows = []
for j in range(N_OUT):
    p = int(cut_rng.integers(0, N_LONG))
    s = int(cut_rng.integers(0, n_days - WIN + 1))
    seg = long_paths[p][s:s + WIN]
    cols[f'sp500_path_{j}'] = seg[:, 0]
    cols[f'dgs10_path_{j}'] = seg[:, 1]
    meta_rows.append({'path_id': j, 'long_seed': SEED0 + p, 'start_idx': s,
                      'start_date': rt.loc[s, 'Date'], 'end_date': rt.loc[s + WIN - 1, 'Date']})

wide = pd.DataFrame(cols)
wide.to_csv(OUT, index=False)
pd.DataFrame(meta_rows).to_csv(OUT.with_name(OUT.stem + '_meta.csv'), index=False)
print('saved', OUT, wide.shape, flush=True)
