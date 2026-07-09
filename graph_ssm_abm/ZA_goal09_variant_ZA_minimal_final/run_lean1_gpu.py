"""
ZA Round K1: J11 機構の選択的巻き戻し

時代分割の知見 (実データの極端分位は「常に正」でなくレジーム相関の増幅、
プールV字は時代構成の産物) により、フェーズ①の flight/正側κスケールの根拠が弱まった。
J11 の副作用 (SP500 abs1/sq1 膨張、rc_q05 の浅化) を、便益が確認済みの機構だけ
残して解消する。

残す: ハブキャップ (スケールleverage)、st チャネル (リードラグ株先行側)、mr_center 6.5
外す: flight、正側κスケール、rl チャネル、disaster_sigma 補填 (0.044→0.038)

  K01: lean = SCALABLE + hubcap + mr65 (フェーズ①③なし)
  K02: K01 + st1.2 (リードラグ株先行側のみ)
  K03: K02 + rl0.3 (金利先行側を最小限残す試み)
  K04: K02 + dgs10_vol_lambda 0.95→0.90 (金利|Δ|クラスタの独立課題)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_SCALABLE_PARAMS

LEAN = dict(ZA_SCALABLE_PARAMS, universe_hub_cap=32, dgs10_mr_center=6.5,
            dgs10_drift_sigma=1.4e-4)

CONFIGS = {
    'K01_lean': dict(LEAN),
    'K02_st12': dict(LEAN, dgs10_stocktrend_beta=1.2),
    'K03_st12_rl03': dict(LEAN, dgs10_stocktrend_beta=1.2, rate_lag_score_beta=0.3),
    'K04_st12_vl090': dict(LEAN, dgs10_stocktrend_beta=1.2, dgs10_vol_lambda=0.90),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'lean1',
        'J11の選択的巻き戻し。目標: abs1→0.30以下/sq1→0.25、rc_q05→-0.5台、'
        'dec10/スケール便益とll_p20維持。K01=lean、K02=+st、K03=+rl最小、K04=+金利クラスタ。',
    )
