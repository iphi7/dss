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
ROOT=Path('graph_ssm_abm/Q_base6_variant_Q_longrun_stable')
RESULTS=ROOT/'results_gpu_round3'
MEMO=ROOT/'検証メモ.md'
SEEDS=[1,2]

def cfg(**overrides):
    params=dict(
        price_impact=0.050,
        exog_common_sigma=0.0040,
        exog_common_jump_prob=0.006,
        exog_common_jump_sigma=0.035,
        exog_common_clip=0.100,
        realized_vol_lambda=0.985,
        vol_sensitivity_mean=0.80,
        vol_sensitivity_std=0.80,
        wealth_sigma=1.20,
        wealth_vol_corr=1.20,
        participation_vol_power=1.80,
        impact_activity_scale=2.50,
        impact_activity_clip=6.00,
        impact_crash_threshold=1.20,
        impact_crash_scale=2.00,
        impact_crash_power=2.00,
        asym_pi_scale=2.00,
        asym_pi_centered=True,
        down_ewma_decay=0.80,
        stoploss_universal_scale=0.0,
        stoploss_universal_threshold=0.005,
        market_anchor_strength=0.006,
        market_anchor_clip=0.005,
        market_anchor_gap_scale=0.70,
        market_anchor_drift=0.00025,
        score_centering=0.0,
        market_risk_premium_score=0.0,
    )
    params.update(overrides)
    return Config(**params)

CONFIGS={
 'Q9_center07_anchor006': cfg(score_centering=0.70),
 'Q10_center10_anchor006': cfg(score_centering=1.00),
 'Q11_center08_prem01': cfg(score_centering=0.80, market_risk_premium_score=0.010),
 'Q12_center08_lowimpact': cfg(score_centering=0.80, price_impact=0.035, impact_activity_scale=1.50, impact_activity_clip=3.50, impact_crash_scale=1.00),
}

def metric_row(df,label):
    r=summarize_stylized_facts(df,label)
    sp=df['sp500'].astype(float).to_numpy()
    r['absacf3_sp500']=_acf(np.abs(sp),3)
    r['r_acf3_sp500']=_acf(sp,3)
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
    cols=['std_sp500','kurt_sp500','absacf5_sp500','absacf3_sp500','leverage_sp500_lag1_20','mean_sp500','r_acf3_sp500','min_sp500_abs','end_sp500_abs','ann_return_approx']
    out={'label':label,'n_samples':len(rows)}
    for c in cols:
        vals=np.array([r[c] for r in rows if np.isfinite(r.get(c,np.nan))],float)
        out[c+'_mean']=float(vals.mean()) if len(vals) else np.nan
        out[c+'_std']=float(vals.std()) if len(vals) else np.nan
        out[c+'_min']=float(vals.min()) if len(vals) else np.nan
        out[c+'_max']=float(vals.max()) if len(vals) else np.nan
    return out

def append(text):
    with MEMO.open('a',encoding='utf-8') as f:
        f.write(text.rstrip()+'\n\n')

def main():
    print('device',DEVICE)
    hist=pd.read_csv('output.csv')
    n_days=min(15120,len(hist)-2)
    append('## Round3: score centering による長期売りバイアス除去\n\nRound2では市場アンカーだけでは指数崩壊が残った。原因仮説: uncertainty/rate ペナルティが全銘柄へ共通に効き、投資家の注文が長期的に売り側へ傾く。そこで投資家ごとに score の平均を一部差し引き、銘柄間の相対評価を導入。')
    all_roll=[]; all_full=[]; all_agg=[]
    for label,c0 in CONFIGS.items():
        t0=time.time(); rolls=[]; fulls=[]
        append(f'### {label}\n\nparams: score_centering={c0.score_centering}, risk_premium_score={c0.market_risk_premium_score}, price_impact={c0.price_impact}, impact_activity_clip={c0.impact_activity_clip}, asym_pi={c0.asym_pi_scale}')
        for seed in SEEDS:
            print('running',label,seed,flush=True)
            c=Config(**{**asdict(c0),'seed':seed,'n_days':n_days})
            outdir=RESULTS/label/f'seed_{seed}'; outdir.mkdir(parents=True,exist_ok=True)
            gen,firms,investors,aux=simulate_market_gpu(hist,c,device=DEVICE)
            gen.to_csv(outdir/'generated_paths.csv',index=False)
            if seed==SEEDS[0]:
                with open(outdir/'config.json','w',encoding='utf-8') as f: json.dump(aux['config'],f,ensure_ascii=False,indent=2)
            fulls.append(metric_row(gen,f'{label}_seed{seed}_full'))
            rolls.extend(rolling(gen,f'{label}_seed{seed}'))
        agg=aggregate(rolls,label+'_rolling'); agg['elapsed_sec']=time.time()-t0
        all_agg.append(agg); all_roll.extend(rolls); all_full.extend(fulls)
        pd.DataFrame(fulls).to_csv(RESULTS/label/'full_metrics.csv',index=False)
        pd.DataFrame(rolls).to_csv(RESULTS/label/'rolling_metrics.csv',index=False)
        status='崩壊あり' if agg['min_sp500_abs_min']<100 else '崩壊ほぼなし'
        append(pd.DataFrame([agg]).to_markdown(index=False)+f'\n\n考察: {status}. rolling std={agg["std_sp500_mean"]:.4f}, kurt={agg["kurt_sp500_mean"]:.2f}, absacf5={agg["absacf5_sp500_mean"]:.3f}, leverage={agg["leverage_sp500_lag1_20_mean"]:.3f}, ann_return={agg["ann_return_approx_mean"]:.3f}.')
    aggdf=pd.DataFrame(all_agg)
    aggdf.to_csv(ROOT/'comparison_longrun_round3.csv',index=False)
    pd.DataFrame(all_roll).to_csv(ROOT/'rolling_metrics_round3.csv',index=False)
    pd.DataFrame(all_full).to_csv(ROOT/'full_metrics_round3.csv',index=False)
    append('### Round3 総合比較\n\n'+aggdf.to_markdown(index=False))
    print(aggdf[['label','std_sp500_mean','kurt_sp500_mean','absacf5_sp500_mean','leverage_sp500_lag1_20_mean','min_sp500_abs_min','ann_return_approx_mean']].to_string(index=False))

if __name__=='__main__':
    main()
