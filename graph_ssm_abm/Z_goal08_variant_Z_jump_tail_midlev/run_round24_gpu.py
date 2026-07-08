"""
Z_goal08 Round24: DGS10 とのレジーム依存相関の導入

実データの90日ローリング相関 corr(株リターン, 金利変化):
  - 1966-97: 一貫して負 (-0.2〜-0.4)、2000-19: 一貫して正 (+0.2〜+0.5)、
    2020s: ゼロ近傍。符号転換は実質1回、レジームは年単位で持続 (lag250自己相関0.70)
  - 固定の rate_sensitivity では符号反転を再現できない

機構: kappa_t = tanh((center - DGS10水準)/width) で金利水準に応じて符号が切り替わる
  経路A (価格直接): firm_return += beta_A * kappa_t * dgs10_change
  経路B (スコア経由): score += beta_B * kappa_t * dgs10_change * (感応度_i/0.10)
    (score_centering の後に加算 — 全銘柄共通項が centering で消えないように)

評価に追加: 90日ローリング相関の時代別平均 (era1=負時代, era2=正時代, era3=現代)、
分位 (q05/q50/q95)、実データ曲線との RMSE。

  Z93: 経路A beta=0.08
  Z94: 経路A beta=0.12
  Z95: 経路B beta=1.0
  Z96: 経路B beta=2.5
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
RESULTS = ROOT / 'results_gpu_round24'
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


def z50_base(**overrides):
    """重エピソード構造 (Round12 Z50)。"""
    params = dict(
        investor_stress_scale=1.4,
        investor_stress_decay_min=0.90, investor_stress_decay_max=0.988,
        vol_sensitivity_mean=0.65, jump_vol_coupling=0.35,
        exog_common_jump_prob=0.0,
        exog_common_jump2_prob=0.012, exog_common_jump2_sigma=0.020, jump2_df=5.0,
        disaster_prob=0.0006, disaster_sigma=0.055,
        disaster_df=3.0, disaster_decay=0.68, disaster_end=0.10,
        jump_aftershock_scale=32.0, jump_aftershock_decay=0.72,
        asym_pi_scale=1.0, asym_pi_threshold=0.25, asym_pi_sat=0.8,
        down_ewma_decay=0.75,
        price_impact=0.025,
    )
    params.update(overrides)
    return z18_base(**params)


def z59_base(**overrides):
    """Z56 + ジャンプ2階層分離 (メガクラッシュ + 再形状エピソード)。"""
    params = dict(
        obs_momentum_scale=0.0, market_anchor_strength=0.002,
        mega_crash_prob=0.000067, mega_crash_size=0.20,
        disaster_prob=0.0004, disaster_sigma=0.075,
        disaster_df=8.0, disaster_decay=0.80, disaster_end=0.30,
        exog_common_clip=0.24, firm_return_clip=0.24,
    )
    params.update(overrides)
    return z50_base(**params)


def z63_base(**overrides):
    """Z60 (2階層+遅い第2余震) + メガクラッシュのエピソード誘発。"""
    params = dict(
        jump_aftershock2_scale=5.0, jump_aftershock2_decay=0.96,
        mega_triggers_episode=1.2,
    )
    params.update(overrides)
    return z59_base(**params)


def z67_base(**overrides):
    """Z64 (メガ誘発+速い余震) + 長い危機 (decay0.85)。"""
    params = dict(
        jump_aftershock_scale=60.0,
        disaster_decay=0.85,
    )
    params.update(overrides)
    return z63_base(**params)


def z71_base(**overrides):
    """Z68 + プラトー化。"""
    params = dict(
        impact_activity_scale=0.90,
        disaster_plateau=6, disaster_decay=0.78, disaster_sigma=0.065,
    )
    params.update(overrides)
    return z67_base(**params)


def z75_base(**overrides):
    """Z73 (薄め) + plateau8 + 危機の負ドリフト。"""
    params = dict(
        disaster_sigma=0.058, jump_aftershock_scale=45.0,
        disaster_plateau=8, disaster_mu=-0.012,
    )
    params.update(overrides)
    return z71_base(**params)


def z77_base(**overrides):
    """Round19 最終候補 Z77。"""
    params = dict(disaster_sigma=0.048, jump_aftershock_scale=35.0)
    params.update(overrides)
    return z75_base(**params)


def z83_base(**overrides):
    """Z81 (iidノイズ再配分) + 参加率日次ノイズ。"""
    params = dict(
        exog_common_sigma=0.0058, price_impact=0.021,
        participation_noise_sigma=0.35,
    )
    params.update(overrides)
    return z77_base(**params)


def z86_base(**overrides):
    """Round21 最良 Z86。"""
    params = dict(
        asym_pi_scale=1.3, disaster_sigma=0.044, exog_common_jump2_sigma=0.016,
    )
    params.update(overrides)
    return z83_base(**params)


def z92_base(**overrides):
    """Round23 最終候補 Z92。"""
    params = dict(
        exog_common_sigma=0.0064, price_impact=0.019,
        asym_pi_scale=1.6, disaster_sigma=0.040,
    )
    params.update(overrides)
    return z86_base(**params)


CONFIGS = {
    'Z93_price008': z92_base(rate_price_beta=0.08),
    'Z94_price012': z92_base(rate_price_beta=0.12),
    'Z95_score10': z92_base(rate_change_score_beta=1.0),
    'Z96_score25': z92_base(rate_change_score_beta=2.5),
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


def rolling_corr_metrics(df, real_roll=None):
    """90日ローリング相関 corr(sp500, DGS10変化) の指標群。"""
    r = df['sp500'].astype(float).reset_index(drop=True)
    dy = df['DGS10'].astype(float).reset_index(drop=True)
    roll = r.rolling(90).corr(dy)
    v = roll.dropna()
    n = len(roll)
    out = {
        'corr_all': float(np.corrcoef(r, dy)[0, 1]),
        'rc_mean': float(v.mean()), 'rc_q05': float(v.quantile(0.05)),
        'rc_q50': float(v.median()), 'rc_q95': float(v.quantile(0.95)),
        # 時代別 (indexは実データ日付に整列): era1=1966-97(負), era2=2000-19(正), era3=2020s
        'rc_era1': float(roll.iloc[90:8000].mean()),
        'rc_era2': float(roll.iloc[8600:13600].mean()),
        'rc_era3': float(roll.iloc[13700:].mean()),
    }
    if real_roll is not None:
        m = min(len(roll), len(real_roll))
        diff = (roll.iloc[:m] - real_roll.iloc[:m]).dropna()
        out['rc_rmse'] = float(np.sqrt((diff ** 2).mean()))
    return out


def metric_row(df, label, real_roll=None):
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
    r.update(rolling_corr_metrics(df, real_roll))
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
    # DGS10 ローリング相関 (Round24+): 時代別平均と RMSE
    for k, wt in [('rc_era1', 1.2), ('rc_era2', 1.2), ('rc_era3', 0.6)]:
        denom = max(abs(tgt[k]), 0.10)
        vals.append(wt * abs(row[k] - tgt[k]) / denom)
    if 'rc_rmse' in row and not np.isnan(row.get('rc_rmse', np.nan)):
        vals.append(1.5 * row['rc_rmse'] / 0.372)  # 実データのrolling corr std で正規化
    # kurt/std
    vals.append(0.35 * abs(row['kurt_sp500'] - tgt['kurt_sp500']) / max(tgt['kurt_sp500'], 5.0))
    vals.append(0.5 * abs(row['std_sp500'] - tgt['std_sp500']) / max(tgt['std_sp500'], 0.005))
    return float(np.nanmean(vals))


def aggregate(rows, label, tgt):
    out = {'label': label, 'n_samples': len(rows)}
    keys = ['std_sp500', 'kurt_sp500',
            'r_acf1', 'abs_acf1', 'abs_acf5', 'abs_acf20', 'abs_acf60',
            'sq_acf1', 'sq_acf5', 'sq_acf20', 'sq_acf60',
            'abs_q099', 'abs_q0999', 'abs_q10',
            'corr_all', 'rc_era1', 'rc_era2', 'rc_era3', 'rc_rmse', 'rc_q05', 'rc_q95']
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
    real_r = real_df['sp500'].astype(float).reset_index(drop=True)
    real_dy = real_df['DGS10'].astype(float).reset_index(drop=True)
    real_roll = real_r.rolling(90).corr(real_dy)
    tgt = real_metrics(real_df)
    tgt['rc_rmse'] = 0.0
    append(
        '## Round24: DGS10レジーム依存相関の導入\n\n'
        '実データ: 90日相関は高金利期に負(-0.2〜-0.4)、低金利期に正(+0.2〜+0.5)、現代ゼロ近傍。'
        'kappa=tanh((center-金利)/width)で符号切替。\n'
        'Z93/Z94=価格直接(0.08/0.12)、Z95/Z96=スコア経由(1.0/2.5)。'
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
            row = metric_row(gen, f'{label}_seed{seed}', real_roll=real_roll)
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
    aggdf.to_csv(ROOT / 'comparison_round24.csv', index=False)
    fulldf.to_csv(ROOT / 'full_metrics_round24.csv', index=False)
    append('### Round1総合比較\n\n' + aggdf.to_markdown(index=False))
    cols = ['label', 'score', 'std_sp500', 'kurt_sp500', 'abs_acf5', 'sq_acf5', 'rc_era1', 'rc_era2', 'rc_era3', 'rc_rmse',
            'abs_q0999', 'abs_q10',
            'lev_dec10_1d', 'lev_dec10_3d', 'lev_dec10_5d', 'lev_dec10_10d', 'lev_dec10_20d',
            'lev_mid_5d', 'lev_mid_10d']
    print(fulldf[cols].to_string(index=False))
    print('\nAGG')
    agg_cols = ['label', 'score_mean', 'rc_era1_mean', 'rc_era2_mean', 'rc_era3_mean', 'rc_rmse_mean', 'std_sp500_mean', 'kurt_sp500_mean',
                'abs_acf5_mean', 'sq_acf5_mean', 'abs_q0999_mean', 'abs_q10_mean',
                'lev_dec10_3d_mean', 'lev_dec10_5d_mean', 'lev_dec10_10d_mean',
                'lev_dec10_20d_mean', 'lev_mid_5d_mean', 'lev_mid_10d_mean']
    print(aggdf[agg_cols].to_string(index=False))
    # 実データ目標も表示
    print('\nREAL TARGET')
    print({k: round(tgt[k], 4) for k in
           ['std_sp500', 'kurt_sp500', 'abs_acf5', 'sq_acf5', 'abs_q0999', 'abs_q10', 'rc_era1', 'rc_era2', 'rc_era3',
            'lev_dec10_3d', 'lev_dec10_5d', 'lev_dec10_10d', 'lev_dec10_20d',
            'lev_mid_5d', 'lev_mid_10d']})


if __name__ == '__main__':
    main()
