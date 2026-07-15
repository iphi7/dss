"""
ZA Round Z1: γの買い/売り分離 — ファイアセール非対称で leverage を取り戻す

Round Y: asym_pi/stress 強化では leverage がほぼ動かず (-0.141→-0.150、経路飽和)。
γ が暴落時のホエールの投げ売り (leverage の主要増幅源) まで絞っていたのが原因。
実市場: 大口の執行制約は平時は両側に効くが、危機のファイアセールでは大量に投げる。
→ 買いγ0.5 維持 (バブル抑制)、売りγを緩めて leverage を復元。

ベース = Y01 (X02 + asym_pi 2.4 + drift 0.00030)。
  Z01: sell γ 0.25
  Z02: sell γ 0.10
  Z03: sell γ 0.0 (売り無制約)
目標: lev_dec10_5d→-0.21, lev_mid→-0.11。
維持: freq≤0.01, floor≤1.05, max12m≤0.8, worst21 -0.36 前後 (深すぎ注意), std~0.011。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL5_PARAMS

Y01 = dict(ZA_FINAL5_PARAMS, whale_size_power=0.5,
           exog_common_sigma=0.0064, market_vol=0.0045,
           market_anchor_drift=0.00030, asym_pi_scale=2.4)

CONFIGS = {
    'Z01_gs025': dict(Y01, whale_size_power_sell=0.25),
    'Z02_gs010': dict(Y01, whale_size_power_sell=0.10),
    'Z03_gs000': dict(Y01, whale_size_power_sell=0.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix6_1',
        'γ買い/売り分離。目標: lev_dec10→-0.21。維持: freq/floor/max12m/std、'
        'worst21が深くなりすぎないか監視。',
    )
