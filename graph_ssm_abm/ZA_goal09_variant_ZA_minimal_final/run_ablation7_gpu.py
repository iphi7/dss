"""
ZA アブレーション Round F: スケール拡大方向の検証

E06/E07 (縮小: 企業40/投資家30) と対で、拡大方向の挙動を見る:
  F01: 企業数 80→160
  F02: 投資家数 60→120
  F03: 両方 (160/120)

興味:
  - 投資家数を増やすと大数の法則で取引由来ボラが平均化され死ぬか?
    (層化抽出なので母集団の形は同じまま密度だけ上がる)
  - 企業数を増やすと指数の分散が下がる (分散投資効果) → std/kurt への影響
  - スタイライズドファクトのスケール不変性

ベースは ZA-final (D03)。判定: score 0.293 ± ノイズフロア0.01。
計算量は n_inv × n_firms^2 に比例 (F03 は1本あたり数分)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL_PARAMS

CONFIGS = {
    'F01_firms160': dict(ZA_FINAL_PARAMS, n_firms=160),
    'F02_investors120': dict(ZA_FINAL_PARAMS, n_investors=120),
    'F03_both_160_120': dict(ZA_FINAL_PARAMS, n_firms=160, n_investors=120),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'ablation7_scale_up',
        'スケール拡大: F01=企業160、F02=投資家120、F03=両方。'
        '縮小方向 (E06/E07) と合わせてスケール感度を両側から評価する。',
    )
