"""
ZA Round AA: 5年窓内のボラクラスタリング回復 (lag5-20 の持続)

診断 (5年窓分布: 実12チャンク vs 生成200窓):
  abs_acf5 実0.205 vs 生成0.081、sq_acf5 実0.163 vs 生成0.067 — 窓内の持続が半分以下。
  60年集計では時代間ボラ差がACFを底上げして僅差に見えるが、実データは
  「どの5年にもある持続的クラスタ」、モデルは「危機エピソード周辺だけ」。
  Q〜Zでノイズ構成が変化 (iid外生ノイズ比重↑、γで内生持続↓) したため。

対処: 遅い持続の機構を強める (ベース = ZA_FINAL6)。
  AA01: jump_aftershock2_scale 6→10 (半減期46日の遅い余震を増強)
  AA02: aftershock2 10 + participation_noise_sigma 0.35→0.25 (iid希釈を減らす)
  AA03: aftershock2 8 + investor_stress_scale 1.4→1.8 + participation_noise 0.28
目標: abs_acf5→0.29(60y)/0.2(5y窓), sq_acf5→0.23, abs_acf1→0.26。
維持: std~0.011, freq≤0.012, floor≤1.1, lev, kurt~21, worst21~-0.36。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL6_PARAMS

CONFIGS = {
    'AA01_as10': dict(ZA_FINAL6_PARAMS, jump_aftershock2_scale=10.0),
    'AA02_as10_pn25': dict(ZA_FINAL6_PARAMS, jump_aftershock2_scale=10.0,
                           participation_noise_sigma=0.25),
    'AA03_as8_st18_pn28': dict(ZA_FINAL6_PARAMS, jump_aftershock2_scale=8.0,
                               investor_stress_scale=1.8,
                               participation_noise_sigma=0.28),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix6_3',
        '窓内ボラクラ回復。目標: abs_acf5→0.29, sq_acf5→0.23, abs_acf1→0.26。'
        '維持: std/freq/floor/lev/kurt/worst21。',
    )
