"""
ZA Round K3: ハブキャップの水準調整 (スケール選択的キャップ)

K2 の特定: abs1/sq1 膨張の主因はハブキャップ32 (×1 でハブの潜在参加者~60人を
強く縛り、ハブ粒度↑→時価総額加重指数の持続↑)。
K07 (キャップなし) で sq1=0.239 (実0.23) に正常化するが ×4 の leverage 回復を失う。

解決: キャップを「×1 ではほぼ縛らず ×4 では縛る」水準へ。固定絶対キャップは
N が大きいほど相対的に強く効くため、水準を上げるだけでスケール選択性が得られる。

  K08: ×1, cap56
  K09: ×1, cap48
  K10: ×4, cap56 (leverage 回復の確認)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_SCALABLE_PARAMS

K03B = dict(ZA_SCALABLE_PARAMS, dgs10_mr_center=6.5,
            dgs10_drift_sigma=1.4e-4, dgs10_stocktrend_beta=1.2,
            rate_lag_score_beta=0.3)

CONFIGS = {
    'K08_s1_cap56': dict(K03B, universe_hub_cap=56),
    'K09_s1_cap48': dict(K03B, universe_hub_cap=48),
    'K10_s4_cap56': dict(K03B, universe_hub_cap=56,
                         n_investors=240, n_firms=320, n_sectors=32),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'lean3',
        'スケール選択的ハブキャップ。目標: ×1 で sq1→0.24/dec10維持、×4 で dec10→-0.185以上。',
    )
