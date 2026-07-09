"""
ZA Round K4 (最終確認): 統一キャップ48の ×4 検証

K3: ×1 は cap48 (K09) でほぼ完全一致。×4 は cap56 で dec10=-0.170、
cap32 で -0.189 だったので、cap48 は中間の ~-0.18 と予想。実測して確定する。
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
    'K11_s4_cap48': dict(K03B, universe_hub_cap=48,
                         n_investors=240, n_firms=320, n_sectors=32),
}

if __name__ == '__main__':
    run_ablation(CONFIGS, 'lean4', '統一cap48の×4確認。')
