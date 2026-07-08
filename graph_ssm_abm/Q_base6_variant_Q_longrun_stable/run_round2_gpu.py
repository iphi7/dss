
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
RESULTS=ROOT/'results_gpu_round2'
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
        asym_pi_scale=2.50,
        asym_pi_centered=True,
        down_ewma_decay=0.80,
        stoploss_universal_scale=0.0,
        stoploss_universal_threshold=0.005,
        market_anchor_strength=0.0,
        market_anchor_clip=0.004,
        market_anchor_gap_scale=0.70,
        market_anchor_drift=0.00025,
    )
    params.update(overrides)
    return Config(**params)

CONFIGS={
 'Q5_mkt_anchor004': cfg(market_anchor_strength=0.004, market_anchor_clip=0.004),
 'Q6_mkt_anchor008': cfg(market_anchor_strength=0.008, market_anchor_clip=0.006),
 'Q7_anchor006_asym2': cfg(market_anchor_strength=0.006, market_anchor_clip=0.005, asym_pi_scale=2.0),
 'Q8_anchor006_G51like': cfg(market_anchor_strength=0.006, market_anchor_clip=0.005, asym_pi_scale=2.5, stoploss_universal_scale=0.04),
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
    with MEMO.open('a',encoding='utf-8') as f: f.write(text.rstrip()+'\n\n')

def main():
    print('device',DEVICE)
    hist=pd.read_csv('output.csv')
    n_days=min(15120,len(hist)-2)
    append('## Round2: 市場ファンダメンタル・アンカー軽量検証\n\n前回の企業別アンカーは崩壊/爆発を十分止めなかったため、指数水準そのものへの弱い市場アンカーを追加。2 seedで軽量評価。')
    all_roll=[]; all_full=[]; all_agg=[]
    for label,c0 in CONFIGS.items():
        t0=time.time(); rolls=[]; fulls=[]
        append(f'### {label}\n\nparams: market_anchor={c0.market_anchor_strength}, asym_pi={c0.asym_pi_scale}, mktfear={c0.stoploss_universal_scale}')
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
        append(pd.DataFrame([agg]).to_markdown(index=False)+'\n\n考察: '+('崩壊あり。' if agg['min_sp500_abs_min']<1 else '崩壊なし。')+f" rolling std={agg['std_sp500_mean']:.4f}, kurt={agg['kurt_sp500_mean']:.2f}, absacf5={agg['absacf5_sp500_mean']:.3f}, leverage={agg['leverage_sp500_lag1_20_mean']:.3f}.")
    aggdf=pd.DataFrame(all_agg)
    aggdf.to_csv(ROOT/'comparison_longrun_round2.csv',index=False)
    pd.DataFrame(all_roll).to_csv(ROOT/'rolling_metrics_round2.csv',index=False)
    pd.DataFrame(all_full).to_csv(ROOT/'full_metrics_round2.csv',index=False)
    append('### Round2 総合比較\n\n'+aggdf.to_markdown(index=False))
    print(aggdf[['label','std_sp500_mean','kurt_sp500_mean','absacf5_sp500_mean','leverage_sp500_lag1_20_mean','min_sp500_abs_min','ann_return_approx_mean']].to_string(index=False))

if __name__=='__main__': main()
