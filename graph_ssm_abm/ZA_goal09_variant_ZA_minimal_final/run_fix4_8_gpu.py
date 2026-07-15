"""
ZA Round P1: パス全体のレベル微調整 (SP500/DGS10 とも実データより少し過大)

O01(ZA-FINAL3) 診断: SP500 年成長 7.83% (実7.45%, +0.38%/年→終値28%高)、
DGS10 平均水準 6.29 (実5.92)。原因: anchor_drift 0.00034 が過大、
平均回帰中心 mr_center 6.5 が実データ平均5.92より高い引力点。少しだけ下げる。

ベース = O01 (M14 + episode_flight 3.5)。
  P01: drift 0.00032 + mr_center 6.0
  P02: drift 0.00031 + mr_center 5.8
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_LEAN_LM_PARAMS

O01 = dict(ZA_LEAN_LM_PARAMS, market_anchor_drift=0.00034,
           market_anchor_drift_sigma=1e-6, down_linear_coef=0.6,
           dgs10_episode_flight=3.5)

CONFIGS = {
    'P01_d32_mr60': dict(O01, market_anchor_drift=0.00032, dgs10_mr_center=6.0),
    'P02_d31_mr58': dict(O01, market_anchor_drift=0.00031, dgs10_mr_center=5.8),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix4_8',
        'パスのレベル微調整。目標: SP500年成長→7.4%前後、DGS10平均水準→5.9前後、'
        '②③④とACF非劣化。',
    )
