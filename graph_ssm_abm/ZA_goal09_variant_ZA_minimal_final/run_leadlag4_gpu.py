"""
ZA フェーズ③ Round J4 (最終): rl 半減による着地

J3 の診断: abs1/sq1 膨張 (0.41/0.39) の主因は rl チャネル自体
(遅い共通売買圧の分散が |r| の持続を作る)。rcsb/pnoise では効かない。
ll_m7 は -0.22 と目標 (-0.15) を超過しているため、rl を下げて
リードラグを目標に合わせると同時に abs 膨張も減らす。

  J10: st1.0, rl0.8, ds0.044
  J11: st1.2, rl0.6, ds0.044
  J12: st1.0, rl0.8, ds0.046, pnoise0.45
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_PHASE2_PARAMS

CONFIGS = {
    'J10_rl08': dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=1.0,
                     rate_lag_score_beta=0.8, disaster_sigma=0.044),
    'J11_rl06_st12': dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=1.2,
                          rate_lag_score_beta=0.6, disaster_sigma=0.044),
    'J12_rl08_pn45': dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=1.0,
                          rate_lag_score_beta=0.8, disaster_sigma=0.046,
                          participation_noise_sigma=0.45),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'leadlag4',
        'rl半減で着地。目標: ll_m7→-0.15、abs1→0.35以下、kurt→16+、他維持。',
    )
