"""
ZA スケール検証 Round G1: 投資ユニバース制限によるスケール不変性

設計:
  - ユニバース = 専門セクター全社 + グラフ隣接 + ランダム5社、上限キャップ36
  - スケール変更時は N/n 比 (0.75) とセクターサイズ (n/n_sectors=10) を固定
  - 銘柄あたり参加者数 ≈ 26人 が全スケールで一定 (キャップ導入後の実測)
    → 不均衡の粒度 sd(I_j) ~ sqrt(H_j) が保存されるはず

  G01: N=30,  n=40,  S=4  (半分)
  G02: N=60,  n=80,  S=8  (基準)
  G03: N=120, n=160, S=16 (2倍)
  G04: N=240, n=320, S=32 (4倍; 1本あたり数分)

比較対象: v2 (ユニバースなし, 60/80) = 0.291、スケール拡大の旧結果 F02/F03 (崩壊 0.42-0.48)。
ユニバース導入自体の影響は G02 vs v2 で読む。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL_V2_PARAMS

UNI = dict(universe_mode='sector_graph', universe_random_extra=5, universe_max_size=36)

CONFIGS = {
    'G01_s30_40': dict(ZA_FINAL_V2_PARAMS, **UNI, n_investors=30, n_firms=40, n_sectors=4),
    'G02_s60_80': dict(ZA_FINAL_V2_PARAMS, **UNI, n_investors=60, n_firms=80, n_sectors=8),
    'G03_s120_160': dict(ZA_FINAL_V2_PARAMS, **UNI, n_investors=120, n_firms=160, n_sectors=16),
    'G04_s240_320': dict(ZA_FINAL_V2_PARAMS, **UNI, n_investors=240, n_firms=320, n_sectors=32),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'scale1_universe',
        'ユニバース制限 (sector+graph+rand5, cap36) + N/n比固定 + セクターサイズ固定での'
        'スケール掃引。銘柄あたり参加者≈26人一定。'
        'G01=半分、G02=基準、G03=2倍、G04=4倍。v2(ユニバースなし)=0.291、旧F02/F03=0.42-0.48。',
    )
