"""
最終モデル (ZA-FINAL7) で 1966年初期値からの60年パスを複数seed生成する。

出力構造 (results/histinit/):
  60y_paths/generated_paths_seedNN.csv  … 60年パス (日次リターン+水準)
  5y_paths/generated_paths_seedNN-C.csv … 60年パスを12個の連続チャンクに分割した約5年窓
  60y_summary.csv / split_metadata.csv  … 概要とチャンクの索引
  config_seedNN.json                    … 使用した全パラメータ

使い方:
  python generate_histinit.py                 # seed 1-10
  python generate_histinit.py --seeds 1 2 3   # seed指定
  python generate_histinit.py --csv /path/to/real.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Config                      # noqa: E402
from model_gpu import simulate_market_gpu     # noqa: E402
from za_final_config import ZA_FINAL7_PARAMS  # noqa: E402

HERE = Path(__file__).resolve().parent
COLS = ['path_id', 'Date', 'sp500_abs', 'DGS10_abs', 'sp500', 'DGS10']


def chunk_bounds(n, k=12):
    base, rem = divmod(n, k)
    bounds, s = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        bounds.append((s, s + size))
        s += size
    return bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='/home/u00121/output.csv',
                    help='実データCSV (Date, sp500_abs, DGS10_abs, sp500, DGS10)')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--out', type=Path, default=HERE / 'results' / 'histinit')
    args = ap.parse_args()

    hist = pd.read_csv(args.csv)
    n_days = len(hist) - 2
    real_tail = hist.tail(n_days).reset_index(drop=True)
    init_sp = float(real_tail.loc[0, 'sp500_abs'])
    init_dg = float(real_tail.loc[0, 'DGS10_abs'])
    dates = real_tail['Date'].to_numpy()
    bounds = chunk_bounds(n_days)

    path60 = args.out / '60y_paths'; path60.mkdir(parents=True, exist_ok=True)
    path5 = args.out / '5y_paths'; path5.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} n_days={n_days} init_sp={init_sp:.4f} init_dg={init_dg:.4f}', flush=True)

    summary_rows, meta_rows = [], []
    for seed in args.seeds:
        lbl = f'seed{seed:02d}'
        print(f'generating {lbl}', flush=True)
        cfg = Config(**{**ZA_FINAL7_PARAMS, 'seed': seed, 'n_days': n_days,
                        'initial_sp500_abs': init_sp, 'initial_dgs10_abs': init_dg})
        paths, _firms, _inv, aux = simulate_market_gpu(hist, cfg, device=device)
        paths = paths.copy()
        paths['Date'] = dates
        paths['path_id'] = 0
        paths = paths[COLS]
        paths.to_csv(path60 / f'generated_paths_{lbl}.csv', index=False)
        with open(args.out / f'config_{lbl}.json', 'w', encoding='utf-8') as f:
            json.dump(aux['config'], f, ensure_ascii=False, indent=2, default=str)
        summary_rows.append({
            'seed': seed, 'file': f'60y_paths/generated_paths_{lbl}.csv', 'rows': len(paths),
            'start_date': paths['Date'].iloc[0], 'end_date': paths['Date'].iloc[-1],
            'sp500_start': paths['sp500_abs'].iloc[0], 'sp500_end': paths['sp500_abs'].iloc[-1],
            'dgs10_start': paths['DGS10_abs'].iloc[0], 'dgs10_end': paths['DGS10_abs'].iloc[-1],
        })
        for ci, (a, b) in enumerate(bounds, start=1):
            ch = paths.iloc[a:b].reset_index(drop=True)
            ch.to_csv(path5 / f'generated_paths_{lbl}-{ci}.csv', index=False)
            meta_rows.append({
                'seed': seed, 'seed_label': lbl, 'chunk': ci, 'rows': len(ch),
                'start_date': ch['Date'].iloc[0], 'end_date': ch['Date'].iloc[-1],
                'path': f'5y_paths/generated_paths_{lbl}-{ci}.csv',
            })

    pd.DataFrame(summary_rows).to_csv(args.out / '60y_summary.csv', index=False)
    pd.DataFrame(meta_rows).to_csv(args.out / 'split_metadata.csv', index=False)
    print('DONE', args.out, flush=True)


if __name__ == '__main__':
    main()
