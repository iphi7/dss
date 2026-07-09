"""
ZA_goal09 アブレーション共通モジュール。

評価指標・スコアは Z_goal08 Round27-30 と同一 (Z117 の score ≈ 0.28-0.30 と直接比較可能)。
各アブレーション腕は dict(Z117_PARAMS, <key>=<off値>) で定義する。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from metrics import summarize_stylized_facts, _acf
from model import Config
from model_gpu import simulate_market_gpu

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path(__file__).parent
MEMO = ROOT / '検証メモ.md'
SEEDS = [1, 2, 3]
LAGS = [1, 2, 3, 5, 10, 20, 60]
LEV_WINDOWS = [1, 3, 4, 5, 7, 10, 20]
MID_DECILES = [5, 6, 7, 8, 9]


def leverage_deciles(r, windows=LEV_WINDOWS, n_deciles=10):
    s = pd.Series(r).dropna().reset_index(drop=True)
    vol = s.abs()
    dec = np.ceil(vol.rank(pct=True) * n_deciles).clip(1, n_deciles).astype(int)
    out = {}
    for w in windows:
        fv = vol.rolling(w).mean().shift(-w)
        for d in range(1, n_deciles + 1):
            idx = dec[dec == d].index
            common = idx.intersection(fv.dropna().index)
            out[f'lev_dec{d}_{w}d'] = (
                float(np.corrcoef(s.iloc[common], fv.iloc[common])[0, 1])
                if len(common) > 5 else np.nan
            )
    return out


def rolling_corr_metrics(df):
    """90日ローリング相関の分布 + DGS10自体の統計。"""
    r = df['sp500'].astype(float).reset_index(drop=True)
    dy = df['DGS10'].astype(float).reset_index(drop=True)
    y = df['DGS10_abs'].astype(float).reset_index(drop=True)
    roll = r.rolling(90).corr(dy)
    v = roll.dropna()
    return {
        'corr_all': float(np.corrcoef(r, dy)[0, 1]),
        'rc_mean': float(v.mean()), 'rc_q05': float(v.quantile(0.05)),
        'rc_q50': float(v.median()), 'rc_q95': float(v.quantile(0.95)),
        'rc_pos_share': float((v > 0).mean()),
        'dy_std': float(dy.std()), 'dy_kurt': float(dy.kurt()),
        'ady_acf1': float(dy.abs().autocorr(1)),
        'y_min': float(y.min()), 'y_max': float(y.max()),
        'y_acf250': float(y.autocorr(250)),
    }


def corr_asym_metrics(df):
    """notebook 7-9節: 相関の非対称性・ボラ連動・リードラグ (7d窓 / 20dMA)。"""
    sp = df['sp500'].astype(float).reset_index(drop=True)
    dg = df['DGS10'].astype(float).reset_index(drop=True)
    out = {}
    corr7 = sp.rolling(7).corr(dg)
    # 7節: 符号分位
    lo1, hi99 = sp.quantile(0.01), sp.quantile(0.99)
    q25, q75 = sp.quantile(0.25), sp.quantile(0.75)
    out['ca_ext_lo'] = float(corr7[sp <= lo1].mean())
    out['ca_ext_hi'] = float(corr7[sp >= hi99].mean())
    out['ca_mid'] = float(corr7[(sp >= q25) & (sp < q75)].mean())
    out['ca_vshape'] = 0.5 * (out['ca_ext_lo'] + out['ca_ext_hi']) - out['ca_mid']
    # 8節: |r| 20分位のホッケースティック
    vol = sp.abs()
    q20 = np.ceil(vol.rank(pct=True) * 20).clip(1, 20).astype(int)
    d20 = float(corr7[q20[q20 == 20].index].mean())
    d10 = float(corr7[q20[q20 == 10].index].mean())
    out['ca_dec20'] = d20
    out['ca_dec10'] = d10
    out['ca_hockey'] = d20 - d10
    # 9節: リードラグ (20dMA)
    sp_ma = sp.rolling(20).mean().dropna().to_numpy()
    dg_ma = dg.rolling(20).mean().dropna().to_numpy()
    m = min(len(sp_ma), len(dg_ma))
    sp_ma, dg_ma = sp_ma[:m], dg_ma[:m]
    for lag, key in [(-7, 'll_m7'), (0, 'll_0'), (20, 'll_p20'), (45, 'll_p45')]:
        if lag > 0:
            out[key] = float(np.corrcoef(sp_ma[:-lag], dg_ma[lag:])[0, 1])
        elif lag < 0:
            out[key] = float(np.corrcoef(sp_ma[-lag:], dg_ma[:lag])[0, 1])
        else:
            out[key] = float(np.corrcoef(sp_ma, dg_ma)[0, 1])
    out['ll_asym'] = out['ll_p20'] - out['ll_m7']  # 実データ ≈ +0.10 - (-0.15) = +0.25
    return out


def metric_row(df, label):
    r = summarize_stylized_facts(df, label)
    sp = df['sp500'].astype(float).to_numpy()
    a = np.abs(sp)
    sq = sp * sp
    for lag in LAGS:
        r[f'r_acf{lag}'] = _acf(sp, lag)
        r[f'abs_acf{lag}'] = _acf(a, lag)
        r[f'sq_acf{lag}'] = _acf(sq, lag)
    for q in [0.95, 0.99, 0.995, 0.999, 1.0]:
        r[f'abs_q{str(q).replace(".", "")}'] = float(np.quantile(a, q))
    r['ann_return_approx'] = float(
        (df['sp500_abs'].iloc[-1] / max(df['sp500_abs'].iloc[0], 1e-12)) ** (252 / len(df)) - 1
    )
    r.update(leverage_deciles(sp))
    for w in LEV_WINDOWS:
        vals = [r.get(f'lev_dec{d}_{w}d', np.nan) for d in MID_DECILES]
        r[f'lev_mid_{w}d'] = float(np.nanmean(vals))
    r.update(rolling_corr_metrics(df))
    r.update(corr_asym_metrics(df))
    return r


def real_metrics(hist):
    return metric_row(hist.copy(), 'real')


def score(row, tgt):
    """Z_goal08 Round27-30 と同一のスコア (低いほど良い)。"""
    vals = []
    for w in LEV_WINDOWS:
        k = f'lev_dec10_{w}d'
        denom = max(abs(tgt[k]), 0.05)
        vals.append(2.2 * abs(row[k] - tgt[k]) / denom)
    for d in MID_DECILES:
        for w in [3, 5, 10]:
            k = f'lev_dec{d}_{w}d'
            denom = max(abs(tgt[k]), 0.05)
            vals.append(0.6 * abs(row[k] - tgt[k]) / denom)
    for w in LEV_WINDOWS:
        k = f'lev_dec1_{w}d'
        denom = max(abs(tgt[k]), 0.05)
        vals.append(0.8 * abs(row[k] - tgt[k]) / denom)
    for kind, wt in [('r_acf', 0.45), ('abs_acf', 1.0), ('sq_acf', 1.0)]:
        for lag in LAGS:
            k = f'{kind}{lag}'
            denom = max(abs(tgt[k]), 0.05)
            vals.append(wt * abs(row[k] - tgt[k]) / denom)
    for k, wt in [('abs_q099', 0.5), ('abs_q0995', 0.6), ('abs_q0999', 0.9), ('abs_q10', 0.5)]:
        vals.append(wt * abs(row[k] - tgt[k]) / max(tgt[k], 1e-4))
    for k, wt in [('rc_q05', 1.0), ('rc_q50', 0.6), ('rc_q95', 1.0), ('rc_pos_share', 0.8)]:
        denom = max(abs(tgt[k]), 0.10)
        vals.append(wt * abs(row[k] - tgt[k]) / denom)
    vals.append(1.0 * abs(row['dy_std'] - tgt['dy_std']) / max(tgt['dy_std'], 0.01))
    vals.append(0.5 * abs(row['dy_kurt'] - tgt['dy_kurt']) / max(tgt['dy_kurt'], 2.0))
    vals.append(0.8 * abs(row['ady_acf1'] - tgt['ady_acf1']) / max(tgt['ady_acf1'], 0.05))
    vals.append(0.5 * abs(row['y_acf250'] - tgt['y_acf250']) / max(tgt['y_acf250'], 0.3))
    vals.append(0.35 * abs(row['kurt_sp500'] - tgt['kurt_sp500']) / max(tgt['kurt_sp500'], 5.0))
    vals.append(0.5 * abs(row['std_sp500'] - tgt['std_sp500']) / max(tgt['std_sp500'], 0.005))
    return float(np.nanmean(vals))


def aggregate(rows, label, tgt):
    out = {'label': label, 'n_samples': len(rows)}
    keys = ['std_sp500', 'kurt_sp500',
            'r_acf1', 'abs_acf1', 'abs_acf5', 'abs_acf20', 'abs_acf60',
            'sq_acf1', 'sq_acf5', 'sq_acf20', 'sq_acf60',
            'abs_q099', 'abs_q0999', 'abs_q10',
            'corr_all', 'rc_q05', 'rc_q50', 'rc_q95', 'rc_pos_share',
            'dy_std', 'dy_kurt', 'ady_acf1', 'y_min', 'y_max', 'y_acf250',
            'ca_ext_lo', 'ca_ext_hi', 'ca_mid', 'ca_vshape', 'ca_dec20', 'ca_dec10', 'ca_hockey',
            'll_m7', 'll_0', 'll_p20', 'll_p45', 'll_asym']
    for d in [1, 10]:
        for w in LEV_WINDOWS:
            keys.append(f'lev_dec{d}_{w}d')
    for w in LEV_WINDOWS:
        keys.append(f'lev_mid_{w}d')
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in rows], float)
        out[k + '_mean'] = float(np.nanmean(vals))
        out[k + '_std'] = float(np.nanstd(vals))
    scores = np.array([score(r, tgt) for r in rows])
    out['score_mean'] = float(np.nanmean(scores))
    out['score_std'] = float(np.nanstd(scores))
    return out


def append_memo(text):
    with MEMO.open('a', encoding='utf-8') as f:
        f.write(text.rstrip() + '\n\n')


def run_ablation(configs: dict, round_name: str, memo_intro: str):
    """configs: {label: params_dict}。結果を results_<round_name>/ と CSV に保存。"""
    print('device', DEVICE)
    hist = pd.read_csv('/home/u00121/output.csv')
    n_days = min(15120, len(hist) - 2)
    real_df = hist.tail(n_days).reset_index(drop=True)
    tgt = real_metrics(real_df)
    results_dir = ROOT / f'results_{round_name}'
    append_memo(f'## {round_name}\n\n{memo_intro}')

    all_rows = []
    aggs = []
    for label, params in configs.items():
        rows = []
        t0 = time.time()
        for seed in SEEDS:
            print('running', label, seed, flush=True)
            c = Config(**{**params, 'seed': seed, 'n_days': n_days})
            outdir = results_dir / label / f'seed_{seed}'
            outdir.mkdir(parents=True, exist_ok=True)
            gen, firms, investors, aux = simulate_market_gpu(hist, c, device=DEVICE)
            gen.to_csv(outdir / 'generated_paths.csv', index=False)
            if seed == SEEDS[0]:
                with open(outdir / 'config.json', 'w', encoding='utf-8') as f:
                    json.dump(aux['config'], f, ensure_ascii=False, indent=2)
            row = metric_row(gen, f'{label}_seed{seed}')
            row['score'] = score(row, tgt)
            rows.append(row)
            all_rows.append(row)
        agg = aggregate(rows, label, tgt)
        agg['elapsed_sec'] = time.time() - t0
        aggs.append(agg)
        pd.DataFrame(rows).to_csv(results_dir / label / 'full_metrics.csv', index=False)
        append_memo(pd.DataFrame([agg]).to_markdown(index=False))

    aggdf = pd.DataFrame(aggs)
    fulldf = pd.DataFrame(all_rows)
    aggdf.to_csv(ROOT / f'comparison_{round_name}.csv', index=False)
    fulldf.to_csv(ROOT / f'full_metrics_{round_name}.csv', index=False)
    append_memo(f'### {round_name} 総合比較\n\n' + aggdf.to_markdown(index=False))

    cols = ['label', 'score_mean', 'std_sp500_mean', 'kurt_sp500_mean',
            'abs_acf1_mean', 'abs_acf5_mean', 'sq_acf1_mean', 'sq_acf5_mean',
            'abs_q0999_mean', 'lev_dec10_5d_mean', 'lev_dec10_10d_mean', 'lev_mid_5d_mean',
            'rc_q05_mean', 'rc_q95_mean', 'dy_std_mean']
    print(aggdf[cols].to_string(index=False))
    print('\nREAL TARGET')
    print({k: round(tgt[k], 4) for k in
           ['std_sp500', 'kurt_sp500', 'abs_acf1', 'abs_acf5', 'sq_acf1', 'sq_acf5',
            'abs_q0999', 'lev_dec10_5d', 'lev_dec10_10d', 'lev_mid_5d',
            'rc_q05', 'rc_q95', 'dy_std']})
    return aggdf
