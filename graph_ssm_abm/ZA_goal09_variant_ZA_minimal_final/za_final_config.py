"""
ZA 最終簡素化モデル (D03) の確定パラメータ。

Z117 から5機構を除去し、残存機構3パラメータを再調整したもの。
score 0.293 (Z117: 0.282、seedノイズフロア ±0.01 内で同等)。

除去した機構 (アブレーション A/B 系で寄与なしと判定):
  - 全員一律 fear          (stoploss_universal_scale=0)
  - ジャンプのボラ連動     (jump_vol_coupling=0)
  - 金利トレンド補正       (rate_trend_scale=0)
  - メガクラッシュのエピソード誘発 (mega_triggers_episode=0)
  - 遅い第2余震            (jump_aftershock2_scale=0)

再調整 (C/D 系; 除去による kurt/sq_acf5 の劣化を回復):
  - disaster_sigma:        0.040 → 0.046
  - jump_aftershock:       (35, 0.72) → (30, 0.80)
  - disaster_plateau:      8 → 10
"""
from z117_config import Z117_PARAMS

ZA_FINAL_PARAMS = dict(
    Z117_PARAMS,
    # ---- 除去 (5機構) ----
    stoploss_universal_scale=0.0,
    jump_vol_coupling=0.0,
    rate_trend_scale=0.0,
    mega_triggers_episode=0.0,
    jump_aftershock2_scale=0.0,
    # ---- 再調整 (3パラメータ) ----
    disaster_sigma=0.046,
    jump_aftershock_scale=30.0,
    jump_aftershock_decay=0.80,
    disaster_plateau=10,
)


# ---- v2: E系で追加確認された除去 (旧流動性クラッシュ項、Δscore=-0.002 で無寄与) ----
# impact_crash_threshold / impact_crash_power も死にパラメータになる。
ZA_FINAL_V2_PARAMS = dict(
    ZA_FINAL_PARAMS,
    impact_crash_scale=0.0,
)


# ---- スケール可変版 (Round G): 投資ユニバース制限 + 再校正 ----
# 銘柄あたり参加者数 ≈26人 が保存され、×1〜×2 でスケール不変
# (×0.5 は粒度過剰、×4 は leverage が緩やかに弱化 — 検証レポート Round G 参照)。
# スケール変更時は n_investors/n_firms = 0.75 と n_firms/n_sectors = 10 を保つこと。
ZA_SCALABLE_PARAMS = dict(
    ZA_FINAL_V2_PARAMS,
    universe_mode="sector_graph",
    universe_random_extra=5,
    universe_max_size=36,
    disaster_sigma=0.038,
    exog_common_jump2_sigma=0.014,
    mega_crash_size=0.18,
    dgs10_sigma0=0.030,
    dgs10_drift_sigma=1.0e-4,
)


# ---- 相関非対称版 (Round H): 7-8節のV字/ホッケー構造 ----
# 状態変数型flight (危機週全体で株安→金利低下)、正側κの非対称スケール、
# 金利の平均回帰中心を実データ平均付近へ。極端分位の相関が mid から
# +0.15〜+0.3 持ち上がる形状を作る (絶対水準は金利パスのレジーム構成に依存)。
ZA_CORR_PARAMS = dict(
    ZA_SCALABLE_PARAMS,
    dgs10_stock_beta=2.5,
    dgs10_flight_beta=2.5,
    dgs10_flight_thresh=2.2,
    dgs10_flight_decay=0.80,
    rate_kappa_pos_scale=0.55,
    dgs10_mr_center=6.5,
    dgs10_drift_sigma=1.4e-4,
)


# ---- フェーズ2確定版 (Round I): 相関非対称 + ハブ参加キャップ ----
ZA_PHASE2_PARAMS = dict(
    ZA_CORR_PARAMS,
    universe_hub_cap=32,
)


# ---- フェーズ3確定版 (Round J): リードラグ導入後の最終形 ----
# 3課題ループの最終到達点 (詳細は検証レポート Round H/I/J):
#   ① 相関の非対称性 (V字/ホッケー) ② スケールでの leverage 維持 ③ リードラグ
# オープン課題: abs_acf1/sq_acf1 が 0.36/0.39 (実 0.26/0.23) に膨張
# (flight/レジーム強化 + rl チャネルの遅い共通売買圧が |r| の持続を作る)。
ZA_PHASE3_PARAMS = dict(
    ZA_PHASE2_PARAMS,
    dgs10_stocktrend_beta=1.2,
    rate_lag_score_beta=0.6,
    disaster_sigma=0.044,
)


# ---- 最終形 ZA-LEAN (Round K): 選択的巻き戻し + スケール選択的ハブキャップ ----
# 時代分割の知見 (7-8節の実データ負ベースは時代平均化の産物、極端分位は
# レジーム相関の増幅) により flight/正側κスケール/rl大を撤去し、
# 便益確認済みの機構のみ残した最終構成。
#   ×1: sq1=0.252 abs1=0.286 dec10=-0.216/-0.200 rc_q05=-0.551 ll=-0.175/+0.090 kurt=24.6
#   ×4: dec10=-0.217/-0.188 (完全回復) sq1=0.25 q999=0.068
# ハブキャップ48は「×1ではほぼ縛らず×4では縛る」スケール選択性を持つ。
ZA_LEAN_PARAMS = dict(
    ZA_SCALABLE_PARAMS,
    universe_hub_cap=48,
    dgs10_mr_center=6.5,
    dgs10_drift_sigma=1.4e-4,
    dgs10_stocktrend_beta=1.2,
    rate_lag_score_beta=0.3,
)


# ---- 長期記憶版 ZA-LEAN-LM (Round L): ACF全形状の改善 ----
# ACF形状診断 (lag1だけ尖り lag20-60 の裾がない) への対応:
#   - 第2余震の復活+持続化 (6, 0.985; 半減期46日) — SP500 の lag2-20
#   - 金利の水準依存ボラ強化 (γ 0.35→0.85) — DGS10 |Δy| の全lag底上げ
#     (副産物: dy_std=0.073, dy_kurt=7.5 が実データ水準に)
#   - 投資家別ストレス記憶上限 0.988→0.995
# 残: lag20-60 の裾は実データの55-60% (完全な長期記憶には多時間スケール
# カスケードが必要 — オープン課題)。
ZA_LEAN_LM_PARAMS = dict(
    ZA_LEAN_PARAMS,
    jump_aftershock2_scale=6.0,
    jump_aftershock2_decay=0.985,
    investor_stress_decay_max=0.995,
    dgs10_vol_gamma=0.85,
)


# ---- 4課題対応版 ZA-FINAL3 (Round M/N/O): 終値・低分位leverage・極端相関 ----
# ④ 終値バイアス: anchor_drift 0.00025→0.00034 (実成長7.7%へ、危機ドリフト込み校正)
#    + パスごと確率ドリフト (σ=1e-6, 専用rng — 共有rngだと共通乱数法が壊れる)
# ③ 低中分位leverage: down_linear_coef=0.6 (閾値なしの弱い線形応答の裾)
#    → dec3/4 = -0.042/-0.042 (実 -0.041/-0.041)
# ② 極端分位の相関: dgs10_episode_flight=3.5 — 危機エピソード中だけ正結合。
#    レジーム符号付き増幅(N1)は「危機の時期配置がseed運」で不成立、
#    エピソード限定flightが正解 (1987型: 高金利時代の暴落でも金利急落)
# ① 中心質量: 未解決 (静穏フロアは exog+個別+粒度+参加ノイズの合成で
#    単一ノブでは削れない)。P(|r|<0.2%)=0.15 vs 実0.23 — オープン課題
ZA_FINAL3_PARAMS = dict(
    ZA_LEAN_LM_PARAMS,
    market_anchor_drift=0.00034,
    market_anchor_drift_sigma=1e-6,
    down_linear_coef=0.6,
    dgs10_episode_flight=3.5,
)
