"""
最終モデル (ZA-FINAL7) で「5年 (1260営業日) × 1000パス」の wide 形式 CSV を作る。

手順:
  1. 60年パスを N_LONG 本生成 (1966年初期値、結果は results/60y_pool/ にキャッシュ)
  2. (パス, 開始位置) を一様乱数で選び、1260日窓を N_OUT 個切り出す
  3. wide 形式 (step, date, sp500_path_i, dgs10_path_i) の1つのCSVに整形
     - sp500_path_i: 日次単純リターン
     - dgs10_path_i: 金利水準の日次差分 (%pt)
  4. 各窓の由来 (元パスのseed・切り出し期間) を *_meta.csv に記録

使い方:
  python generate_5y_1000paths.py
  python generate_5y_1000paths.py --n-long 30 --n-out 1000 --out my_paths.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Config                      # noqa: E402
from model_gpu import simulate_market_gpu     # noqa: E402
from za_final_config import ZA_FINAL7_PARAMS  # noqa: E402

HERE = Path(__file__).resolve().parent
WIN = 1260


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='/home/u00121/output.csv')
    ap.add_argument('--n-long', type=int, default=30, help='60年パスの本数')
    ap.add_argument('--n-out', type=int, default=1000, help='切り出す5年窓の数')
    ap.add_argument('--seed0', type=int, default=3001, help='60年パスのseed起点')
    ap.add_argument('--cut-seed', type=int, default=424242, help='窓抽出rngのseed')
    ap.add_argument('--out', type=Path,
                    default=HERE / 'results' / 'synthetic_returns_final7_1000paths_wide.csv')
    args = ap.parse_args()

    hist = pd.read_csv(args.csv)
    n_days = len(hist) - 2
    rt = hist.tail(n_days).reset_index(drop=True)
    init_sp = float(rt.loc[0, 'sp500_abs'])
    init_dg = float(rt.loc[0, 'DGS10_abs'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pool_dir = HERE / 'results' / '60y_pool'
    pool_dir.mkdir(parents=True, exist_ok=True)
    long_paths = []
    for i in range(args.n_long):
        seed = args.seed0 + i
        f = pool_dir / f'long_seed{seed}.csv'
        if f.exists():
            g = pd.read_csv(f)
        else:
            print(f'generating 60y path {i+1}/{args.n_long} (seed {seed})', flush=True)
            c = Config(**{**ZA_FINAL7_PARAMS, 'seed': seed, 'n_days': n_days,
                          'initial_sp500_abs': init_sp, 'initial_dgs10_abs': init_dg})
            g, _, _, _ = simulate_market_gpu(hist, c, device=device)
            g = g[['sp500', 'DGS10']].astype(np.float32)
            g.to_csv(f, index=False)
        long_paths.append(g[['sp500', 'DGS10']].to_numpy(dtype=np.float32))
    print('60y pool ready', flush=True)

    cut_rng = np.random.default_rng(args.cut_seed)
    # date列: 実データ最終日の翌営業日から1260日 (SigGAN形式と同じ構造)
    last_date = pd.to_datetime(hist['Date'].iloc[-1])
    dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=WIN)
    cols = {'step': np.arange(WIN), 'date': dates.strftime('%Y-%m-%d')}
    meta_rows = []
    for j in range(args.n_out):
        p = int(cut_rng.integers(0, args.n_long))
        s = int(cut_rng.integers(0, n_days - WIN + 1))
        seg = long_paths[p][s:s + WIN]
        cols[f'sp500_path_{j}'] = seg[:, 0]
        cols[f'dgs10_path_{j}'] = seg[:, 1]
        meta_rows.append({'path_id': j, 'long_seed': args.seed0 + p, 'start_idx': s,
                          'start_date': rt.loc[s, 'Date'],
                          'end_date': rt.loc[s + WIN - 1, 'Date']})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(args.out, index=False)
    pd.DataFrame(meta_rows).to_csv(args.out.with_name(args.out.stem + '_meta.csv'), index=False)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
