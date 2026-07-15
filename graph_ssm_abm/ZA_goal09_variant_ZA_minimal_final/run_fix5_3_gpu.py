"""
ZA Round S: バブル規模・頻度の解消 = ベースノイズ削減 (最終配合の確定)

Round R で trend_weight は無関係と判明 (3倍下げても不変)。真因は日次ボラ過大:
  Q04(exog0.0050/mvol0.0037) で std 1.08%(実1.05)・max12m +76%(実+77)・
  freq>50% 2.95→1.29% と実データへ。ただし Q04 は危機も削り kurt 15.0(実21.8)。
狙い: ノイズ削減だけを効かせ、危機(尖度源)は温存して kurt を保ちつつバブルを消す。

ベース = ZA_FINAL4 (Q01: disaster_mu-0.007/plateau6)。
  S01: exog 0.0050 + mvol 0.0037 (強B、危機は温存)
  S02: exog 0.0054 + mvol 0.0040 (中B、kurt寄り)
  S03: exog 0.0050 + mvol 0.0037 + disaster_mu -0.006 + mega 0.15 (= Q04相当)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL4_PARAMS

CONFIGS = {
    'S01_Bstrong': dict(ZA_FINAL4_PARAMS, exog_common_sigma=0.0050, market_vol=0.0037),
    'S02_Bmid': dict(ZA_FINAL4_PARAMS, exog_common_sigma=0.0054, market_vol=0.0040),
    'S03_Q04': dict(ZA_FINAL4_PARAMS, exog_common_sigma=0.0050, market_vol=0.0037,
                    disaster_mu=-0.006, mega_crash_size=0.15),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix5_3',
        'ベースノイズ削減でバブル解消。目標: std→0.0105, max_12m_up→0.77, '
        'freq_12m_gt50→0.006方向, kurt維持(>17), abs/sq_acf1・lev_dec10・ca_vshape非劣化。',
    )
