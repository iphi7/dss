"""
ZA スケール検証 Round G4: 金利生成側の再校正

G3 の結果: 株側は完成形 (G09: kurt20.2, dec10完璧, rc_q95正常化)。
score を押し上げるのは金利側:
  - dy_std 0.048 (目標0.067)、水準レンジ狭小 (y_min 2.4)、5.5%以下に滞留
    → κ>0 優勢で rc_q50 が +0.13 に反転 (実 -0.13)
  - ユニバース化で株→金利チャネル入力が変化した影響

対応: 金利の確率ショックとドリフトを増強して放浪を回復。
  G13: ×1 dampA + rate(σ0 0.024→0.030, drift_σ 8e-5→1.0e-4)
  G14: ×2 同上
  G15: ×1 dampA + rate(σ0 0.032, drift_σ 1.2e-4, mr_theta 0.00025→0.0002)
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
RATE_2 = dict(dgs10_sigma0=0.032, dgs10_drift_sigma=1.2e-4, dgs10_mr_theta=0.0002)

CONFIGS = {
    'G13_s1_rate1': dict(ZA_FINAL_V2_PARAMS, **UNI, **DAMP_A, **RATE_1,
                         n_investors=60, n_firms=80, n_sectors=8),
    'G14_s2_rate1': dict(ZA_FINAL_V2_PARAMS, **UNI, **DAMP_A, **RATE_1,
                         n_investors=120, n_firms=160, n_sectors=16),
    'G15_s1_rate2': dict(ZA_FINAL_V2_PARAMS, **UNI, **DAMP_A, **RATE_2,
                         n_investors=60, n_firms=80, n_sectors=8),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'scale4_rate',
        '金利側の再校正 (dy_std 0.048→0.067, 水準放浪の回復, rc_q50反転の解消)。'
        'G13=×1 rate1、G14=×2 rate1、G15=×1 rate2(強め)。',
    )
