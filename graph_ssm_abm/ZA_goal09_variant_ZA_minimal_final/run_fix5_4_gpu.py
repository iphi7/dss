"""
ZA Round T: 平穏ボラ床の経年ラチェットを止める (暴騰暴落の頻度偏在の根治)

時代別分解で判明:
  日次変化率std: 実 0.82→1.22(+48%)  FINAL5 0.84→1.29(+53%) … 全体は近い
  >50%上昇/年の頻度: 実 0.84%→0.79% (時代一定)  FINAL5 0.00%→3.23% (激しく偏在)
  → 原因は平穏ボラ床の経年ラチェット(vol_floor_drift ~1.35)。前半は床が低く
    暴騰0%、直近は床が持ち上がり3.2%。ノイズ一律削減(S01)では偏在は直らない。
修正: realized_var_ewma に固定アンカーへの平均回帰を追加し、ボラEWMAの正
  フィードバックによる床ドリフトを抑制。前半頻度↑・直近頻度↓で実の一定へ。

ベース = ZA_FINAL5 (S01)。監視: sq_acf1/sq_acf5(クラスタ)・abs_acf1・kurt・
  lev_dec10・std を壊さないアンカー強度を選定。
  T01: anchor 0.011, pull 0.010
  T02: anchor 0.011, pull 0.030
  T03: anchor 0.011, pull 0.060
  T04: anchor 0.010, pull 0.030
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL5_PARAMS

CONFIGS = {
    'T01_a11_p010': dict(ZA_FINAL5_PARAMS, realized_vol_anchor=0.011, realized_vol_anchor_pull=0.010),
    'T02_a11_p030': dict(ZA_FINAL5_PARAMS, realized_vol_anchor=0.011, realized_vol_anchor_pull=0.030),
    'T03_a11_p060': dict(ZA_FINAL5_PARAMS, realized_vol_anchor=0.011, realized_vol_anchor_pull=0.060),
    'T04_a10_p030': dict(ZA_FINAL5_PARAMS, realized_vol_anchor=0.010, realized_vol_anchor_pull=0.030),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix5_4',
        'ボラ床ラチェット抑制。目標: vol_floor_drift→1.0, freq_12m_gt50→0.006, '
        '時代偏在の解消。監視: sq_acf1/sq_acf5・abs_acf1・kurt・lev_dec10・std非劣化。',
    )
