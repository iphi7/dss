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

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT=Path('graph_ssm_abm/R_base7_variant_R_acf_control')
RESULTS=ROOT/'results_gpu_round3'
MEMO=ROOT/'検証メモ.md'
SEEDS=[1,2]
TARGET=dict(std=0.01053,kurt=21.75,absacf5=0.289,sqacf5=0.228,racf1=-0.015,lev=-0.046,ann=0.077)

def cfg(**overrides):
    params=dict(
        # Q12/Q17 周辺を土台に、ACF過剰要因を弱める
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

CONFIGS={
 'R09_rebalance005': cfg(idio_vol=0.0070, exog_common_sigma=0.0038, exog_common_jump_prob=0.006, exog_common_jump_sigma=0.035, participation_vol_power=1.10, impact_activity_scale=1.10, impact_activity_clip=2.50, asym_pi_scale=1.40, down_ewma_decay=0.65, stoploss_universal_scale=0.030, portfolio_rebalance_rate=0.005, portfolio_cash_target=0.05),
 'R10_rebalance010': cfg(idio_vol=0.0070, exog_common_sigma=0.0038, exog_common_jump_prob=0.006, exog_common_jump_sigma=0.035, participation_vol_power=1.10, impact_activity_scale=1.10, impact_activity_clip=2.50, asym_pi_scale=1.40, down_ewma_decay=0.65, stoploss_universal_scale=0.030, portfolio_rebalance_rate=0.010, portfolio_cash_target=0.05),
 'R11_rebal_activity': cfg(idio_vol=0.0070, exog_common_sigma=0.0038, exog_common_jump_prob=0.006, exog_common_jump_sigma=0.035, participation_vol_power=1.20, impact_activity_scale=1.30, impact_activity_clip=2.80, asym_pi_scale=1.50, down_ewma_decay=0.65, stoploss_universal_scale=0.040, stoploss_universal_threshold=0.003, portfolio_rebalance_rate=0.010, portfolio_cash_target=0.05),
}

def metric_row(df,label):
    r=summarize_stylized_facts(df,label)
    sp=df['sp500'].astype(float).to_numpy()
    for lag in [1,2,3,5,10,20,60]:
        r[f'r_acf{lag}']=_acf(sp,lag)
        r[f'abs_acf{lag}']=_acf(np.abs(sp),lag)
        r[f'sq_acf{lag}']=_acf(sp**2,lag)
    r['min_sp500_abs']=float(df['sp500_abs'].min())
    r['end_sp500_abs']=float(df['sp500_abs'].iloc[-1])
    r['ann_return_approx']=float((df['sp500_abs'].iloc[-1]/max(df['sp500_abs'].iloc[0],1e-12))**(252/len(df))-1)
    return r

def rolling(df,label,window=1260):
    rows=[]
    for start in range(0,len(df)-window+1,window):
        rr=metric_row(df.iloc[start:start+window].reset_index(drop=True),f'{label}_w{start//window:02d}')
        rr['window']=start//window
        rows.append(rr)
    return rows

def aggregate(rows,label):
    cols=['std_sp500','kurt_sp500','abs_acf5','sq_acf5','r_acf1','r_acf5','leverage_sp500_lag1_20','skew_sp500','min_sp500_abs','ann_return_approx']
    out={'label':label,'n_samples':len(rows)}
    for c in cols:
        vals=np.array([r[c] for r in rows if np.isfinite(r.get(c,np.nan))],float)
        out[c+'_mean']=float(vals.mean()) if len(vals) else np.nan
        out[c+'_std']=float(vals.std()) if len(vals) else np.nan
        out[c+'_min']=float(vals.min()) if len(vals) else np.nan
        out[c+'_max']=float(vals.max()) if len(vals) else np.nan
    return out

def full_score(row):
    return (
        abs(row['r_acf1']-TARGET['racf1'])/0.05
        + abs(row['abs_acf5']-TARGET['absacf5'])/0.20
        + abs(row['sq_acf5']-TARGET['sqacf5'])/0.20
        + abs(row['std_sp500']-TARGET['std'])/0.010
        + abs(row['leverage_sp500_lag1_20']-TARGET['lev'])/0.06
        + abs(row['ann_return_approx']-TARGET['ann'])/0.15
    )

def append(text):
    with MEMO.open('a',encoding='utf-8') as f: f.write(text.rstrip()+'\n\n')

def main():
    print('device',DEVICE)
    hist=pd.read_csv('output.csv')
    n_days=min(15120,len(hist)-2)
    append('## Round3: 投資家turnoverで全期間ACFの長期レジームを抑える\n\nRound2ではrolling ACFは低い一方、seedによって全期間ACFが高くなった。長期の保有/現金固定化がレジームを作る仮説から、弱いportfolio rebalancingを追加して検証する。')
    all_roll=[]; all_full=[]; all_agg=[]
    for label,c0 in CONFIGS.items():
        t0=time.time(); rolls=[]; fulls=[]
        append(f'### {label}\n\nparams: momentum={c0.momentum_score_weight}, part_power={c0.participation_vol_power}, impact_scale={c0.impact_activity_scale}, down_decay={c0.down_ewma_decay}, fear={c0.stoploss_universal_scale}')
        for seed in SEEDS:
            print('running',label,seed,flush=True)
            c=Config(**{**asdict(c0),'seed':seed,'n_days':n_days})
            outdir=RESULTS/label/f'seed_{seed}'; outdir.mkdir(parents=True,exist_ok=True)
            gen,firms,investors,aux=simulate_market_gpu(hist,c,device=DEVICE)
            gen.to_csv(outdir/'generated_paths.csv',index=False)
            if seed==SEEDS[0]:
                with open(outdir/'config.json','w',encoding='utf-8') as f: json.dump(aux['config'],f,ensure_ascii=False,indent=2)
            fr=metric_row(gen,f'{label}_seed{seed}_full'); fr['score']=full_score(fr); fulls.append(fr)
            rolls.extend(rolling(gen,f'{label}_seed{seed}'))
        agg=aggregate(rolls,label+'_rolling'); agg['elapsed_sec']=time.time()-t0; agg['full_score_mean']=float(np.mean([x['score'] for x in fulls]))
        all_agg.append(agg); all_roll.extend(rolls); all_full.extend(fulls)
        pd.DataFrame(fulls).to_csv(RESULTS/label/'full_metrics.csv',index=False)
        pd.DataFrame(rolls).to_csv(RESULTS/label/'rolling_metrics.csv',index=False)
        append(pd.DataFrame([agg]).to_markdown(index=False)+f'\n\nfull mean score={agg["full_score_mean"]:.3f}.')
    aggdf=pd.DataFrame(all_agg); fulldf=pd.DataFrame(all_full)
    aggdf.to_csv(ROOT/'comparison_round3.csv',index=False)
    fulldf.to_csv(ROOT/'full_metrics_round3.csv',index=False)
    pd.DataFrame(all_roll).to_csv(ROOT/'rolling_metrics_round3.csv',index=False)
    append('### Round3 総合比較\n\n'+aggdf.to_markdown(index=False))
    print(fulldf[['label','std_sp500','kurt_sp500','r_acf1','abs_acf5','sq_acf5','leverage_sp500_lag1_20','min_sp500_abs','ann_return_approx','score']].to_string(index=False))
    print('\nAGG')
    print(aggdf[['label','std_sp500_mean','kurt_sp500_mean','r_acf1_mean','abs_acf5_mean','sq_acf5_mean','leverage_sp500_lag1_20_mean','min_sp500_abs_min','ann_return_approx_mean','full_score_mean']].to_string(index=False))

if __name__=='__main__': main()
