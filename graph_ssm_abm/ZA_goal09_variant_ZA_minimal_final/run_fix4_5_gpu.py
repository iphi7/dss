"""
ZA Round N1: ② 極端日のレジーム符号付き増幅

実データの時代内構造「高ボラほどその時代のレジーム相関が増幅」
(負時代: mid-0.27→d20-0.39、正時代: +0.26→+0.55) を再現する。
旧 flight (常に正) と違い kappa の符号を保つため、負レジーム seed では
極端分位がより負に、正レジーム seed ではより正になるはず。

ベース = M14 (③④確定形)。
  N01: amp 1.5
  N02: amp 3.0
  N03: amp 3.0, thresh 1.5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_LEAN_LM_PARAMS

M14 = dict(ZA_LEAN_LM_PARAMS, market_anchor_drift=0.00034,
           market_anchor_drift_sigma=1e-6, down_linear_coef=0.6)

CONFIGS = {
    'N01_amp15': dict(M14, dgs10_regime_amp=1.5),
    'N02_amp30': dict(M14, dgs10_regime_amp=3.0),
    'N03_amp30_t15': dict(M14, dgs10_regime_amp=3.0, dgs10_amp_thresh=1.5),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix4_5',
        'レジーム符号付き増幅。目標: 時代内リフト |ext|-|mid| が両符号で発生 '
        '(seed別のca_ext vs ca_midで判定)、rc_q05/dy_std非劣化。',
    )
