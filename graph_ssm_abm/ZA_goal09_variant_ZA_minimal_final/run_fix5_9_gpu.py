"""
ZA Round Y: γ導入で弱まった leverage の補償 (最終調整2)

Round X: γ0.5+ノイズ完全復元(X02) で std/freq/床/worst21/ACF が実データ水準。
残: lev_dec10 -0.141 (実-0.209)・lev_mid -0.040 (実-0.109) — γ がホエールの
暴落時大口売り (leverageの増幅源の一部) も絞ったため。
→ leverage機構を強めて補償。ann 0.079→0.0745 へ drift も微調整。

ベース = X02 (ZA_FINAL5 + γ0.5 + exog0.0064/mvol0.0045) + drift 0.00030。
  Y01: asym_pi_scale 1.6→2.4
  Y02: investor_stress_scale 1.4→2.1
  Y03: asym_pi 2.0 + stress 1.8 (併用)
目標: lev_dec10→-0.21, lev_mid→-0.11。維持: std~0.0105, freq≤0.01,
floor≤1.05, abs1~0.26, sq1~0.22, worst21~-0.36。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL5_PARAMS

X02 = dict(ZA_FINAL5_PARAMS, whale_size_power=0.5,
           exog_common_sigma=0.0064, market_vol=0.0045,
           market_anchor_drift=0.00030)

CONFIGS = {
    'Y01_pi24': dict(X02, asym_pi_scale=2.4),
    'Y02_st21': dict(X02, investor_stress_scale=2.1),
    'Y03_pi20_st18': dict(X02, asym_pi_scale=2.0, investor_stress_scale=1.8),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix5_9',
        'γで弱まったleverageの補償。目標: lev_dec10→-0.21, lev_mid→-0.11。'
        '維持: std/freq/floor/abs1/sq1/worst21。',
    )
