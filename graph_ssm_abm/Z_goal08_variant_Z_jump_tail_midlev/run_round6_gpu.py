"""
Z_goal08 Round6: 有効要素の統合 + ジャンプのボラ連動

Round5 の結果:
  - Z24 (stress=1.4): seed2 の mid_5d=-0.095 (目標-0.109に接近)
  - Z25 (hetero記憶): seed3 の ACF/kurt が改善
  - Z26 (tail): q999=0.082 と目標(0.070)超え。ジャンプはやや弱めて統合する
  - 未解決: seed1 の部分発火。静穏期に大ジャンプが乗り kurt が 78-89 に爆発

Round6 は統合 + ジャンプサイズのボラ連動 (jump_vol_coupling):
  ジャンプ振幅 ×= clip((実現ボラ/基準ボラ)^c, 0.3, 3.0)。
  静穏期の不自然な巨大ジャンプを抑え (kurt正常化)、活発期の tail は保つ。

  Z27: Z18 + stress1.4 + hetero記憶 + tail(0.0017,0.115) — 統合のみ
  Z28: Z27 + jump_vol_coupling=0.5
  Z29: Z27 + jump_vol_coupling=1.0
  Z30: Z28 + vol_sensitivity_mean=0.65
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import Config
from model_gpu import simulate_market_gpu
from metrics import summarize_stylized_facts, _acf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path('graph_ssm_abm/Z_goal08_variant_Z_jump_tail_midlev')
RESULTS = ROOT / 'results_gpu_round6'
MEMO = ROOT / '検証メモ.md'
SEEDS = [1, 2, 3]
LAGS = [1, 2, 3, 5, 10, 20, 60]
LEV_WINDOWS = [1, 3, 4, 5, 7, 10, 20]
MID_DECILES = [5, 6, 7, 8, 9]


def cfg(**overrides):
    # Y14/Y15 の共通ベース (Y round3/4 の cfg と同一)
    params = dict(
        price_impact=0.030,
        idio_vol=0.0060,
        exog_common_sigma=0.0030,
        exog_common_jump_prob=0.004,
        exog_common_jump_sigma=0.030,
        exog_common_clip=0.080,
        realized_vol_lambda=0.970,
        vol_sensitivity_mean=0.55,
        vol_sensitivity_std=0.65,
        wealth_sigma=1.00,
        wealth_vol_corr=0.80,
        participation_vol_power=0.90,
        impact_activity_scale=0.70,
        impact_activity_clip=2.00,
        impact_crash_threshold=1.25,
        impact_crash_scale=0.30,
        impact_crash_power=2.00,
        asym_pi_scale=1.20,
        asym_pi_centered=True,
        down_ewma_decay=0.55,
        stoploss_universal_scale=0.015,
        stoploss_universal_threshold=0.004,
        market_anchor_strength=0.006,
        market_anchor_clip=0.005,
        market_anchor_gap_scale=0.70,
        market_anchor_drift=0.00025,
        score_centering=0.80,
        market_risk_premium_score=0.0,
        momentum_score_weight=0.00,
        firm_return_clip=0.08,
    )
    params.update(overrides)
    return Config(**params)


def z01_base(**overrides):
    """Y14/Y15 の中間点。"""
    params = dict(
        idio_vol=0.0080, exog_common_sigma=0.00445,
        exog_common_jump_prob=0.0015, exog_common_jump_sigma=0.105,
        exog_common_clip=0.120,
        rare_shock_prob=0.0040, rare_shock_sigma=0.28,
        participation_vol_power=1.21, impact_activity_scale=1.40, impact_activity_clip=3.20,
        asym_pi_scale=0.40, down_ewma_decay=0.60,
        stoploss_universal_scale=0.0005, stoploss_universal_threshold=0.006,
        portfolio_rebalance_rate=0.0008,
        investor_stress_scale=1.20, investor_stress_decay=0.9575,
        investor_stress_threshold=0.0, investor_stress_clip=3.35,
        risk_pref_participation_scale=0.45, risk_pref_size_scale=0.30,
        risk_averse_withdraw_scale=0.29,
    )
    params.update(overrides)
    return cfg(**params)


def z07_combo(**overrides):
    """Round1 の複合案 (余震 + midlev) をベースにする。"""
    params = dict(
        firm_return_clip=0.20, exog_common_clip=0.20,
        jump_aftershock_scale=20.0, jump_aftershock_decay=0.55,
        asym_pi_scale=0.55, asym_pi_threshold=0.55, asym_pi_sat=1.3,
    )
    params.update(overrides)
    return z01_base(**params)


def z18_base(**overrides):
    params = dict(
        market_vol=0.0045, realized_vol_lambda=0.95,
        vol_factor_max=2.2, participation_cap=2.8,
        investor_stratified=True,
    )
    params.update(overrides)
    return z07_combo(**params)


def z27_combo(**overrides):
    params = dict(
        investor_stress_scale=1.4,
        investor_stress_decay_min=0.90, investor_stress_decay_max=0.988,
        exog_common_jump_prob=0.0017, exog_common_jump_sigma=0.115,
    )
    params.update(overrides)
    return z18_base(**params)


CONFIGS = {
    'Z27_combo': z27_combo(),
    'Z28_combo_vc05': z27_combo(jump_vol_coupling=0.5),
    'Z29_combo_vc10': z27_combo(jump_vol_coupling=1.0),
    'Z30_vc05_vs065': z27_combo(jump_vol_coupling=0.5, vol_sensitivity_mean=0.65),
}


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
    # 中位分位 (5-9) の平均 leverage (表示・比較用)
    for w in LEV_WINDOWS:
        vals = [r.get(f'lev_dec{d}_{w}d', np.nan) for d in MID_DECILES]
        r[f'lev_mid_{w}d'] = float(np.nanmean(vals))
    return r


def real_metrics(hist):
    df = hist.copy()
    if 'sp500' not in df.columns and 'SP500' in df.columns:
        df = df.rename(columns={'SP500': 'sp500', 'SP500_abs': 'sp500_abs',
                                'DGS10': 'dgs10', 'DGS10_abs': 'dgs10_abs'})
    return metric_row(df, 'real')


def score(row, tgt):
    """Y round4 の score に、中位分位 leverage と tail 分位点を追加した拡張版。"""
    vals = []
    # decile10 multi-window (最重要)
    for w in LEV_WINDOWS:
        k = f'lev_dec10_{w}d'
        denom = max(abs(tgt[k]), 0.05)
        vals.append(2.2 * abs(row[k] - tgt[k]) / denom)
    # 中位分位 (5-9): Y15で薄まりすぎた問題 (ユーザー指摘)
    for d in MID_DECILES:
        for w in [3, 5, 10]:
            k = f'lev_dec{d}_{w}d'
            denom = max(abs(tgt[k]), 0.05)
            vals.append(0.6 * abs(row[k] - tgt[k]) / denom)
    # decile1 の異常な正/負相関も抑える
    for w in LEV_WINDOWS:
        k = f'lev_dec1_{w}d'
        denom = max(abs(tgt[k]), 0.05)
        vals.append(0.8 * abs(row[k] - tgt[k]) / denom)
    # multi-lag ACF
    for kind, wt in [('r_acf', 0.45), ('abs_acf', 1.0), ('sq_acf', 1.0)]:
        for lag in LAGS:
            k = f'{kind}{lag}'
            denom = max(abs(tgt[k]), 0.05)
            vals.append(wt * abs(row[k] - tgt[k]) / denom)
    # tail 分位点: ジャンプサイズ不足 (ユーザー指摘, Q-Qプロット対応)
    for k, wt in [('abs_q099', 0.5), ('abs_q0995', 0.6), ('abs_q0999', 0.9), ('abs_q10', 0.5)]:
        vals.append(wt * abs(row[k] - tgt[k]) / max(tgt[k], 1e-4))
    # kurt/std
    vals.append(0.35 * abs(row['kurt_sp500'] - tgt['kurt_sp500']) / max(tgt['kurt_sp500'], 5.0))
    vals.append(0.5 * abs(row['std_sp500'] - tgt['std_sp500']) / max(tgt['std_sp500'], 0.005))
    return float(np.nanmean(vals))


def aggregate(rows, label, tgt):
    out = {'label': label, 'n_samples': len(rows)}
    keys = ['std_sp500', 'kurt_sp500',
            'r_acf1', 'abs_acf1', 'abs_acf5', 'abs_acf20', 'abs_acf60',
            'sq_acf1', 'sq_acf5', 'sq_acf20', 'sq_acf60',
            'abs_q099', 'abs_q0999', 'abs_q10']
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


def append(text):
    with MEMO.open('a', encoding='utf-8') as f:
        f.write(text.rstrip() + '\n\n')


def main():
    print('device', DEVICE)
    hist = pd.read_csv('output.csv')
    n_days = min(15120, len(hist) - 2)
    real_df = hist.tail(n_days).reset_index(drop=True)
    tgt = real_metrics(real_df)
    append(
        '## Round6: 統合 + ジャンプのボラ連動\n\n'
        'Z24/Z25/Z26の有効要素を統合し、jump_vol_couplingで静穏期の巨大ジャンプを抑える。\n'
        'Z27=統合、Z28=+vc0.5、Z29=+vc1.0、Z30=Z28+vs_mean0.65。'
    )
    all_rows = []
    aggs = []
    for label, c0 in CONFIGS.items():
        rows = []
        t0 = time.time()
        append(
            f'### {label}\n\n'
            f'params: frclip={c0.firm_return_clip}, cclip={c0.exog_common_clip}, '
            f'jump=({c0.exog_common_jump_prob},{c0.exog_common_jump_sigma},df={c0.jump_df}), '
            f'aftershock=({c0.jump_aftershock_scale},{c0.jump_aftershock_decay}), '
            f'asym=({c0.asym_pi_scale},thr={c0.asym_pi_threshold},sat={c0.asym_pi_sat}), '
            f'stress=({c0.investor_stress_scale},{c0.investor_stress_decay},'
            f'hetero=[{c0.investor_stress_decay_min},{c0.investor_stress_decay_max}])'
        )
        for seed in SEEDS:
            print('running', label, seed, flush=True)
            c = Config(**{**asdict(c0), 'seed': seed, 'n_days': n_days})
            outdir = RESULTS / label / f'seed_{seed}'
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
        pd.DataFrame(rows).to_csv(RESULTS / label / 'full_metrics.csv', index=False)
        append(pd.DataFrame([agg]).to_markdown(index=False))
    aggdf = pd.DataFrame(aggs)
    fulldf = pd.DataFrame(all_rows)
    aggdf.to_csv(ROOT / 'comparison_round6.csv', index=False)
    fulldf.to_csv(ROOT / 'full_metrics_round6.csv', index=False)
    append('### Round1総合比較\n\n' + aggdf.to_markdown(index=False))
    cols = ['label', 'score', 'std_sp500', 'kurt_sp500', 'abs_acf5', 'sq_acf5',
            'abs_q0999', 'abs_q10',
            'lev_dec10_1d', 'lev_dec10_3d', 'lev_dec10_5d', 'lev_dec10_10d', 'lev_dec10_20d',
            'lev_mid_5d', 'lev_mid_10d']
    print(fulldf[cols].to_string(index=False))
    print('\nAGG')
    agg_cols = ['label', 'score_mean', 'std_sp500_mean', 'kurt_sp500_mean',
                'abs_acf5_mean', 'sq_acf5_mean', 'abs_q0999_mean', 'abs_q10_mean',
                'lev_dec10_3d_mean', 'lev_dec10_5d_mean', 'lev_dec10_10d_mean',
                'lev_dec10_20d_mean', 'lev_mid_5d_mean', 'lev_mid_10d_mean']
    print(aggdf[agg_cols].to_string(index=False))
    # 実データ目標も表示
    print('\nREAL TARGET')
    print({k: round(tgt[k], 4) for k in
           ['std_sp500', 'kurt_sp500', 'abs_acf5', 'sq_acf5', 'abs_q0999', 'abs_q10',
            'lev_dec10_3d', 'lev_dec10_5d', 'lev_dec10_10d', 'lev_dec10_20d',
            'lev_mid_5d', 'lev_mid_10d']})


if __name__ == '__main__':
    main()
