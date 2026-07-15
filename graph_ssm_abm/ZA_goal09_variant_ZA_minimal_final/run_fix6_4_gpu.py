"""
ZA Round AB: 余震増強の副作用 (worst21/q999) を危機一撃の縮小で相殺

AA01 (aftershock2 6→10): ACF全域改善 (abs5 0.17→0.20, abs20 0.055→0.087,
score 0.302) だが余震が暴落後ボラを数週増幅し worst21 -0.394 (実-0.356)・
q999 0.087 (実0.070) に悪化。→ disaster_sigma を絞って深さ/裾を戻す。

ベース = ZA_FINAL6 + jump_aftershock2_scale=10。
  AB01: disaster_sigma 0.044→0.040
  AB02: disaster_sigma 0.042
  AB03: aftershock2 12 + disaster_sigma 0.038 (クラスタさらに強め+一撃さらに絞る)
目標: abs5≥0.20, sq5≥0.15 を維持しつつ worst21→-0.36, q999→0.075, freq≤0.012。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL6_PARAMS

AA01 = dict(ZA_FINAL6_PARAMS, jump_aftershock2_scale=10.0)

CONFIGS = {
    'AB01_ds040': dict(AA01, disaster_sigma=0.040),
    'AB02_ds042': dict(AA01, disaster_sigma=0.042),
    'AB03_as12_ds038': dict(AA01, jump_aftershock2_scale=12.0, disaster_sigma=0.038),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix6_4',
        '余震増強の副作用相殺。目標: abs5≥0.20/sq5≥0.15維持、worst21→-0.36、'
        'q999→0.075、freq≤0.012、lev/床非劣化。',
    )
