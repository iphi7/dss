"""
ZA Round L1: 長期記憶成分の復元/強化

診断 (ACF全形状の実測):
  - SP500 |r|/r² の記憶が数日〜2週の短時定数に集中 (実データは lag60 まで裾がある)。
    アブレーションで除去した第2余震 (2-3週, decay0.96) が長 lag 形状の担い手だった
    可能性 (score が lag 等重みで形状に鈍感だった)
  - DGS10 |Δy| の長期記憶 (実 lag60=0.18) は「水準依存ボラ×単位根水準」由来。
    γ=0.35 + 単一の速い EWMA では不足

  L01: 第2余震の復活 (5, 0.96)
  L02: 第2余震 遅く強く (4, 0.98)
  L03: 投資家別ストレス記憶の上限 0.988→0.995 (株の月スケール記憶)
  L04: dgs10_vol_gamma 0.35→0.65 (金利の水準経由の長期記憶)
  L05: 金利の遅い第2分散成分 (λ2=0.99)
  L06: L01+L03+L04 の複合
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_LEAN_PARAMS

CONFIGS = {
    'L01_after2': dict(ZA_LEAN_PARAMS, jump_aftershock2_scale=5.0, jump_aftershock2_decay=0.96),
    'L02_after2_slow': dict(ZA_LEAN_PARAMS, jump_aftershock2_scale=4.0, jump_aftershock2_decay=0.98),
    'L03_stress995': dict(ZA_LEAN_PARAMS, investor_stress_decay_max=0.995),
    'L04_gamma065': dict(ZA_LEAN_PARAMS, dgs10_vol_gamma=0.65),
    'L05_h2': dict(ZA_LEAN_PARAMS, dgs10_vol_lambda2=0.99),
    'L06_combo': dict(ZA_LEAN_PARAMS, jump_aftershock2_scale=5.0, jump_aftershock2_decay=0.96,
                      investor_stress_decay_max=0.995, dgs10_vol_gamma=0.65),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'longmem1',
        '長期記憶の復元。目標: sp_abs lag20 0.06→0.18方向、sp_sq lag5-20↑、'
        'dy_abs 全lag 0.12→0.25方向。lag1と既存指標の非劣化。',
    )
