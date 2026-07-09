"""
ZA アブレーション Round E: 追加簡素化 + 設計ベースの検証

グループ1 (追加の簡素化候補):
  E01: メガクラッシュ本体の除去 (誘発は除去済みだが本体 p=6.7e-5, -0.20 は未検証)
  E02: 旧流動性クラッシュ項の除去 (impact_crash_scale 0.3 — N/O系の遺産)
  E03: 企業レベル稀ショックの除去 (rare_shock_prob 0.004 — 状態第1次元の t3)

グループ2 (設計ベースの効きの検証):
  E04: グラフなし (graph_topology='zero' — 企業間伝播・投資家グラフ項が全消滅)
  E05: 完全グラフ知識 (subjective_graph_mode='perfect' — 欠損グラフ(情報の不完全性)の効き)
  E06: 企業数 80→40
  E07: 投資家数 60→30

ベースは ZA-final (D03)。判定: score 0.293 ± ノイズフロア0.01。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL_PARAMS

CONFIGS = {
    # --- グループ1: 追加簡素化 ---
    'E01_no_mega': dict(ZA_FINAL_PARAMS, mega_crash_prob=0.0),
    'E02_no_crashterm': dict(ZA_FINAL_PARAMS, impact_crash_scale=0.0),
    'E03_no_rareshock': dict(ZA_FINAL_PARAMS, rare_shock_prob=0.0),
    # --- グループ2: 設計ベース ---
    'E04_zero_graph': dict(ZA_FINAL_PARAMS, graph_topology='zero'),
    'E05_perfect_graph': dict(ZA_FINAL_PARAMS, subjective_graph_mode='perfect'),
    'E06_firms40': dict(ZA_FINAL_PARAMS, n_firms=40),
    'E07_investors30': dict(ZA_FINAL_PARAMS, n_investors=30),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'ablation6_base',
        '追加簡素化 (E01=メガ本体, E02=旧クラッシュ項, E03=企業稀ショック) と'
        '設計ベース検証 (E04=グラフなし, E05=完全グラフ知識, E06=企業40, E07=投資家30)。'
        'ベース=ZA-final(D03) 0.293。',
    )
