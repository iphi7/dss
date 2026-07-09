"""
ZA 相関非対称ループ Round H3: 金利レジーム構成の是正 (mr_center 引き上げ)

H2 の結果: 極端分位は実データ水準到達 (H06: +0.34/+0.25)。
残る mid (+0.09, 目標-0.08) と rc_q05 劣化 (-0.32, 目標-0.57) は同根:
生成金利が強い負レジーム (高金利) に十分滞在しない。
実データは60年中32年が5.5%超 (平均5.92%)。

対応: dgs10_mr_center を実データ平均付近へ (5.5→6.5/7.0)。
kappa中心 (rate_regime_center=5.5) は変えない — 「金利は平均6%前後を漂い、
5.5%を境にレジームが切り替わる」という構造になる。

  H08: H06 + mr_center 6.5
  H09: H06 + mr_center 7.0
  H10: H08 + pos_scale 1.0 (レジーム構成が直れば正側非対称は不要かの切り分け)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_SCALABLE_PARAMS

H06 = dict(ZA_SCALABLE_PARAMS, dgs10_flight_beta=4.0, dgs10_flight_thresh=1.5,
           dgs10_stock_beta=2.0, rate_kappa_pos_scale=0.55)

CONFIGS = {
    'H08_mr65': dict(H06, dgs10_mr_center=6.5),
    'H09_mr70': dict(H06, dgs10_mr_center=7.0),
    'H10_mr65_ps10': dict(H06, dgs10_mr_center=6.5, rate_kappa_pos_scale=1.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'corr3_mrcenter',
        '金利の平均回帰中心を実データ平均へ (負レジーム滞在を回復)。'
        '目標: mid→-0.08、rc_q05→-0.57、rc_q50→-0.13、vshape/hockey維持。'
        'H08=6.5、H09=7.0、H10=6.5+正側対称に戻す。',
    )
