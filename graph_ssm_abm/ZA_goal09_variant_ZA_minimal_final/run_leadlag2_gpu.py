"""
ZA フェーズ③ Round J2: リードラグの強度調整

J1: J02 (st0.4+rl3.0) で符号反転構造が出現 (ll_asym=+0.271 ✓)。調整点:
  - 金利先行側 -0.250 (目標-0.15) → rl 3.0→2.0
  - 株先行側 +0.021 (目標+0.10) → st 0.4→0.7/1.0
  - kurt 19→10.7 の低下 → 危機σで補填 or rl減で自然回復を確認
  - J03 (rl6.0) は r_acf1=+0.15 汚染で棄却

  J04: st0.7, rl2.0
  J05: st0.7, rl2.0, disaster_sigma 0.038→0.042 (kurt補填)
  J06: st1.0, rl1.5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_PHASE2_PARAMS

CONFIGS = {
    'J04_st07_rl20': dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=0.7,
                          rate_lag_score_beta=2.0),
    'J05_st07_rl20_ds42': dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=0.7,
                               rate_lag_score_beta=2.0, disaster_sigma=0.042),
    'J06_st10_rl15': dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=1.0,
                          rate_lag_score_beta=1.5),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'leadlag2',
        'リードラグ強度調整。目標: ll_m7→-0.15、ll_p20→+0.10、r_acf1<0.05、kurt回復。',
    )
