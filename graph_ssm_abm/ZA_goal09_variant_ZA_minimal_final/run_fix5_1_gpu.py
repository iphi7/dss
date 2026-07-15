"""
ZA Round Q: 危機の深さ (週次/月次の過大な累積下落) + ① 中心質量

診断 (P01 histinit 10seed):
  - 連続下落の「日数」は実データと一致 (最長10-14日 vs 実12)。長すぎではない。
  - 問題は危機発火時の「深さ」: 最悪5日 -20〜-54%(実-29)、最悪21日 -28〜-53%(実-36)。
    disaster_mu(-0.012) × plateau(10) + 幾何減衰 で 3-4週間 毎日-1.2%の持続ドリフト
    → 月次の深い下落。これが図の「一気に落ちる」の正体。
  - ① 平穏期でも std が実超え(66-85で0.8-1.2% vs 実0.82)。exog_common/market_vol が高い。

ベース = ZA_FINAL3 (P01)。
  A(危機の深さ): disaster_mu, disaster_plateau, mega_crash_size を弱める
  B(中心質量):   exog_common_sigma, market_vol を下げる (副産物: abs/sq_acf1 も実へ)

  Q01 = A のみ (mu -0.007, plateau 6)
  Q02 = B のみ (exog 0.0054, mvol 0.0039)
  Q03 = A + B
  Q04 = A + B 強め (mu -0.006, mega 0.15, exog 0.0050, mvol 0.0037)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL3_PARAMS

A = dict(disaster_mu=-0.007, disaster_plateau=6)
B = dict(exog_common_sigma=0.0054, market_vol=0.0039)

CONFIGS = {
    'Q01_depthA': dict(ZA_FINAL3_PARAMS, **A),
    'Q02_centerB': dict(ZA_FINAL3_PARAMS, **B),
    'Q03_AB': dict(ZA_FINAL3_PARAMS, **A, **B),
    'Q04_AB_strong': dict(ZA_FINAL3_PARAMS,
                          disaster_mu=-0.006, disaster_plateau=6, mega_crash_size=0.15,
                          exog_common_sigma=0.0050, market_vol=0.0037),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix5_1',
        '危機の深さ(worst_5d/21d→実-29/-36)と①中心質量(p_lt002→0.23, std→0.0105)。'
        '監視: sq_acf20/60(余震由来の長期記憶)・q999・lev_dec10・vshape非劣化。',
    )
