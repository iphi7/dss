"""
ZA-FINAL4 (Q01: 危機の深さ修正版) の histinit 10seed 生成。

O01_ef35/histinit と同じ初期値・命名規則・ディレクトリ構造で再現する:
  - 実データ tail 先頭 (1966-01-05) の水準を初期値に使う historical-init
  - 60y_paths/generated_paths_seedNN.csv (seed 1-10, 14882営業日)
  - 5y_paths/generated_paths_seedNN-C.csv (12個の連続 disjoint ~5年チャンク)
  - 60y_summary.csv / split_metadata.csv / config_seedNN.json
モデルパラメータのみ ZA_FINAL3 (P01: drift0.00032 / mr_center6.0) に更新。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ZA = Path('/home/u00121/graph_ssm_abm/ZA_goal09_variant_ZA_minimal_final')
sys.path.insert(0, str(ZA))
from model import Config                     # noqa: E402
from model_gpu import simulate_market_gpu    # noqa: E402
from za_final_config import ZA_FINAL4_PARAMS  # noqa: E402

OUTROOT = ZA / 'results_fix5_1' / 'Q01_final' / 'histinit'
PATH60 = OUTROOT / '60y_paths'
PATH5 = OUTROOT / '5y_paths'
PATH60.mkdir(parents=True, exist_ok=True)
PATH5.mkdir(parents=True, exist_ok=True)

REAL = '/home/u00121/output.csv'
COLS = ['path_id', 'Date', 'sp500_abs', 'DGS10_abs', 'sp500', 'DGS10']


def chunk_bounds(n, k=12):
    """先頭 rem 個を1本多くして n を k 個の連続 disjoint チャンクに割る (O01 と同じ)。"""
    base, rem = divmod(n, k)
    bounds, s = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        bounds.append((s, s + size))
        s += size
    return bounds


def main():
    hist = pd.read_csv(REAL)
    n_days = len(hist) - 2                       # 14882 (O01 と一致)
    real_tail = hist.tail(n_days).reset_index(drop=True)
    init_sp = float(real_tail.loc[0, 'sp500_abs'])
    init_dg = float(real_tail.loc[0, 'DGS10_abs'])
    dates = real_tail['Date'].to_numpy()
    bounds = chunk_bounds(n_days)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} n_days={n_days} init_sp={init_sp:.4f} init_dg={init_dg:.4f}', flush=True)

    summary_rows, meta_rows = [], []
    for seed in range(1, 11):
        lbl = f'seed{seed:02d}'
        print(f'generating {lbl}', flush=True)
        cfg = Config(**{**ZA_FINAL4_PARAMS, 'seed': seed, 'n_days': n_days,
                        'initial_sp500_abs': init_sp, 'initial_dgs10_abs': init_dg})
        paths, _firms, _inv, aux = simulate_market_gpu(hist, cfg, device=device)
        paths = paths.copy()
        paths['Date'] = dates                    # 歴史期間に整列
        paths['path_id'] = 0
        paths = paths[COLS]
        paths.to_csv(PATH60 / f'generated_paths_{lbl}.csv', index=False)

        with open(OUTROOT / f'config_{lbl}.json', 'w', encoding='utf-8') as f:
            json.dump(aux['config'], f, ensure_ascii=False, indent=2, default=str)

        summary_rows.append({
            'seed': seed, 'file': f'60y_paths/generated_paths_{lbl}.csv', 'rows': len(paths),
            'start_date': paths['Date'].iloc[0], 'end_date': paths['Date'].iloc[-1],
            'sp500_start': paths['sp500_abs'].iloc[0], 'sp500_end': paths['sp500_abs'].iloc[-1],
            'dgs10_start': paths['DGS10_abs'].iloc[0], 'dgs10_end': paths['DGS10_abs'].iloc[-1],
        })
        for ci, (a, b) in enumerate(bounds, start=1):
            ch = paths.iloc[a:b].reset_index(drop=True)
            ch.to_csv(PATH5 / f'generated_paths_{lbl}-{ci}.csv', index=False)
            meta_rows.append({
                'seed': seed, 'seed_label': lbl, 'chunk': ci, 'rows': len(ch),
                'start_date': ch['Date'].iloc[0], 'end_date': ch['Date'].iloc[-1],
                'path': f'5y_paths/generated_paths_{lbl}-{ci}.csv',
            })

    pd.DataFrame(summary_rows).to_csv(OUTROOT / '60y_summary.csv', index=False)
    pd.DataFrame(meta_rows).to_csv(OUTROOT / 'split_metadata.csv', index=False)
    print('DONE', OUTROOT, flush=True)


if __name__ == '__main__':
    main()
