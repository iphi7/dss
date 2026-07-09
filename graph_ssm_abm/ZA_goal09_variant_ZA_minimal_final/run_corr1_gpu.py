"""
ZA 相関非対称ループ Round H1: 極端日 flight-to-quality 結合の導入

実データ (notebook 7-9節, 7d窓):
  - 7節 V字: 極端分位 (≤1%: +0.33, ≥99%: +0.24) で正、中間 (25-75%): −0.08
    → ca_vshape = +0.36
  - 8節 ホッケー: |r|20分位の dec20=+0.17 vs dec10=−0.10 → ca_hockey = +0.27
  - 中間分位の弱い負は「負レジーム32年 vs 正20年」の混合平均 (−0.06) とほぼ一致
    → 平常日結合はレジーム型のままで良く、欠けているのは極端日の κ 非依存の正結合
    (1987: 高金利時代でも暴落日は国債買い→金利急低下 = corr 正)

機構: Δy += flight_beta × sign(r) × max(|r| − flight_thresh × 実現ボラ, 0)
  (株由来項なので投資家スコアの exo 成分からは除外される)

  H01: flight (1.5, 1.2)
  H02: flight (3.0, 1.2)
  H03: flight (3.0, 1.5) + 平常日結合 stock_beta 2.5→2.0
  H04: flight (5.0, 1.5)

ベース = ZA_SCALABLE (G13 設定)。判定: ca_vshape/ca_hockey が目標へ、
既存指標 (score, rc分布, dy_std) の非劣化。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_SCALABLE_PARAMS

CONFIGS = {
    'H01_f15_t12': dict(ZA_SCALABLE_PARAMS, dgs10_flight_beta=1.5, dgs10_flight_thresh=1.2),
    'H02_f30_t12': dict(ZA_SCALABLE_PARAMS, dgs10_flight_beta=3.0, dgs10_flight_thresh=1.2),
    'H03_f30_t15_sb20': dict(ZA_SCALABLE_PARAMS, dgs10_flight_beta=3.0,
                             dgs10_flight_thresh=1.5, dgs10_stock_beta=2.0),
    'H04_f50_t15': dict(ZA_SCALABLE_PARAMS, dgs10_flight_beta=5.0, dgs10_flight_thresh=1.5),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'corr1_flight',
        '極端日flight結合の導入 (7-8節のV字/ホッケー再現)。'
        '目標: ca_vshape=+0.36, ca_hockey=+0.27 (実データ)。'
        'H01=弱, H02=中, H03=中+平常結合減, H04=強。',
    )
