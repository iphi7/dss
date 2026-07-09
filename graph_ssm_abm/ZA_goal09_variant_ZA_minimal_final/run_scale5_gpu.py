"""
ZA スケール検証 Round G5 (最終確認): G13設定で全スケールを埋める

G13設定 (ユニバース + 減衰A + 金利再校正1) の ×1/×2 は確認済み (0.331/0.348)。
残る ×0.5 と ×4 を検証してスケール掃引を完成させる。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL_V2_PARAMS

UNI = dict(universe_mode='sector_graph', universe_random_extra=5, universe_max_size=36)
DAMP_A = dict(disaster_sigma=0.038, exog_common_jump2_sigma=0.014, mega_crash_size=0.18)
RATE_1 = dict(dgs10_sigma0=0.030, dgs10_drift_sigma=1.0e-4)

CONFIGS = {
    'G16_s05': dict(ZA_FINAL_V2_PARAMS, **UNI, **DAMP_A, **RATE_1,
                    n_investors=30, n_firms=40, n_sectors=4),
    'G17_s4': dict(ZA_FINAL_V2_PARAMS, **UNI, **DAMP_A, **RATE_1,
                   n_investors=240, n_firms=320, n_sectors=32),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'scale5_confirm',
        'G13設定の残りスケール確認。G16=×0.5 (30/40/4)、G17=×4 (240/320/32)。',
    )
