"""
ZA フェーズ③ Round J3 (最終): 副作用のリバランス

J2 (J06) でリードラグ構造達成 (ll_asym=+0.31, r_acf1=-0.02, dec10=-0.203)。
副作用: 相関/リードラグ機構の蓄積で金利→スコア共通項の分散が増え、
abs_acf1/sq_acf1 が 0.40/0.37 に膨張 (実 0.26/0.23)、kurt 12.8 に低下。

対応: 共通スコア分散の削減 + 危機σ補填。
  J07: J06 + rate_change_score_beta 0.45→0.30 + disaster_sigma 0.042
  J08: J07 + participation_noise 0.35→0.45
  J09: J06 + rcsb 0.35 + disaster_sigma 0.044 + pnoise 0.45
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_PHASE2_PARAMS

J06 = dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=1.0, rate_lag_score_beta=1.5)

CONFIGS = {
    'J07_rcsb03_ds42': dict(J06, rate_change_score_beta=0.30, disaster_sigma=0.042),
    'J08_pn045': dict(J06, rate_change_score_beta=0.30, disaster_sigma=0.042,
                      participation_noise_sigma=0.45),
    'J09_rcsb035_ds44': dict(J06, rate_change_score_beta=0.35, disaster_sigma=0.044,
                             participation_noise_sigma=0.45),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'leadlag3',
        '副作用リバランス。目標: abs1→0.30以下、sq1→0.25、kurt→18前後、'
        'リードラグ/相関/leverage構造の維持。',
    )
