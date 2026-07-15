"""
ZA Round X: γ確定後のベースノイズ復元 (最終調整)

Round W: キャパシティ制約 γ=0.5 で床ラチェット完全解消 (floor_drift 1.46→0.95)、
バブル頻度 1.7%→0.3% (実0.6%)。ただし市場が大人しくなりすぎ:
std 0.009 (実0.0105)・max12m 0.52 (実0.77)・kurt 41 (ジャンプが相対的に際立つ)・
lev -0.16 (実-0.21)。
→ S01 のノイズ削減はホエール由来の内生ボラ過大への対症だった。γ が根治した今、
  ベースノイズを部分復元して std/kurt/lev を実データへ戻す。

ベース = ZA_FINAL5 + γ0.5。
  X01: exog 0.0057, mvol 0.0041 (部分復元)
  X02: exog 0.0064, mvol 0.0045 (ZA_FINAL4 水準へ完全復元)
  X03: exog 0.0060, mvol 0.0043 (中間)
目標: std→0.0105, kurt→22, lev_dec10→-0.21, sq_acf1→0.23, max12m→0.77。
維持: freq≤0.01, floor_drift≤1.1。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL5_PARAMS

G = dict(ZA_FINAL5_PARAMS, whale_size_power=0.5)

CONFIGS = {
    'X01_e57_m41': dict(G, exog_common_sigma=0.0057, market_vol=0.0041),
    'X02_e64_m45': dict(G, exog_common_sigma=0.0064, market_vol=0.0045),
    'X03_e60_m43': dict(G, exog_common_sigma=0.0060, market_vol=0.0043),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix5_8',
        'γ0.5+ノイズ復元。目標: std→0.0105, kurt→22, lev→-0.21, max12m→0.77, '
        'freq≤0.01, floor_drift≤1.1 維持。',
    )
