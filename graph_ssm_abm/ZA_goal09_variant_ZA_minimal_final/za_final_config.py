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


# ---- 4課題対応版 ZA-FINAL3 (Round M/N/O/P): 終値・低分位leverage・極端相関・レベル ----
# ④ 終値バイアス: anchor_drift (実成長7.45%へ校正)
#    + パスごと確率ドリフト (σ=1e-6, 専用rng — 共有rngだと共通乱数法が壊れる)
# ③ 低中分位leverage: down_linear_coef=0.6 (閾値なしの弱い線形応答の裾)
#    → dec3/4 = -0.042/-0.042 (実 -0.041/-0.041)
# ② 極端分位の相関: dgs10_episode_flight=3.5 — 危機エピソード中だけ正結合。
#    レジーム符号付き増幅(N1)は「危機の時期配置がseed運」で不成立、
#    エピソード限定flightが正解 (1987型: 高金利時代の暴落でも金利急落)
# ① 中心質量: 未解決 (静穏フロアは exog+個別+粒度+参加ノイズの合成で
#    単一ノブでは削れない)。P(|r|<0.2%)=0.15 vs 実0.23 — オープン課題
#
# Round P (レベル微調整): O01(drift0.00034/mr6.5) は histinit で SP500 年成長
# 7.83%(実7.45%)・DGS10 平均水準 6.29(実5.92) と両方やや過大 →
#   drift 0.00034→0.00032 (成長 histinit中央値 ~7.3%、実のわずか下 = 多様性許容域)
#   dgs10_mr_center 6.5→6.0 (平均水準 5.91 で実5.92にほぼ一致)
# 副作用なし: abs/sq_acf・q999・dec10 leverage 不変、②リフトはむしろ +0.08→+0.10
# (低金利滞在が増え flight レジームが明瞭化)。
ZA_FINAL3_PARAMS = dict(
    ZA_LEAN_LM_PARAMS,
    market_anchor_drift=0.00032,
    market_anchor_drift_sigma=1e-6,
    down_linear_coef=0.6,
    dgs10_episode_flight=3.5,
    dgs10_mr_center=6.0,
)


# ---- ZA-FINAL4 (Round Q): 危機の深さ (週次/月次の過大な累積下落) 修正 ----
# histinit の SP500 レベル図で「直近ほど一気に落ちる」大変動の診断:
#   連続下落の「日数」は実データ並み (最長10-14日 vs 実12)。問題は「深さ」。
#   disaster_mu(-0.012) × plateau(10) + 幾何減衰 で 3-4週間 毎日-1.2%×強度 の
#   持続ドリフトがかかり、worst_5d/21d が -20〜-54% / -28〜-53% (実 -29/-36) と過大。
# 修正: disaster_mu -0.012→-0.007、disaster_plateau 10→6 (=A腕)。
#   → worst_5d/21d = -0.290/-0.352 (実 -0.292/-0.356) にほぼ一致。
#   副産物: 過大だった abs/sq_acf1 (0.305/0.276) も実 (0.261/0.230) へ、
#           sq_acf20/60 の長期記憶も実側へ、aggregate score も改善 (0.315→0.298)。
#   コスト: kurt 20.5→15.8 (危機を浅くした直接の帰結。窓依存が大きく許容範囲)。
# ① 中心質量 (p_lt002≈0.15 vs 実0.23) はベースノイズ2割減でも 0.16-0.17 止まりで
#   単一ノブでは動かず、オープン課題として据え置き (B腕は kurt/score とトレードオフ)。
ZA_FINAL4_PARAMS = dict(
    ZA_FINAL3_PARAMS,
    disaster_mu=-0.007,
    disaster_plateau=6,
)


# ---- ZA-FINAL5 (Round R/S): 暴騰暴落の「規模・頻度」過剰 = 日次ボラ過大 ----
# histinit 図の「直近ほど猛烈な上昇/下降」を精査した診断:
#   - 12ヶ月変化率のピーク規模は実データ同等(seed6 +89% vs 実+77%)だが、
#     +50%級の暴騰局面が実データの ~4倍の頻度(2.4% vs 実0.6%)で起きる。
#   - 平穏ボラ床が経年ラチェット上昇(21d std q25: 0.71→1.02、実は横ばい0.6)。
# Round R(棄却): trend_weight×delta のモメンタムが原因という仮説は実験で否定。
#   trend_weight 0.28→0.10 に下げてもバブル指標・vol_floor_drift は不変。
# Round S(確定): 真因は日次ボラ過大。バブル頻度・規模は 12ヶ月リターン分布の
#   広がり(std過大)の帰結。ベースノイズ exog_common_sigma/market_vol を下げると
#   std 0.013→0.012・max_12m +88%→+80%・freq 2.4%→1.7% と実データへ収束。
#   危機機構は温存(=尖度源)したため kurt 19.6 を維持(Q04は危機も削り15.0に低下)。
# 残: 平穏床のラチェット(vol_floor_drift ~1.35)と①中心質量はオープン
#   (ボラEWMAの正フィードバックの構造的性質。全体レベルは実データ近傍に。)
ZA_FINAL5_PARAMS = dict(
    ZA_FINAL4_PARAMS,
    exog_common_sigma=0.0050,
    market_vol=0.0037,
)


# ---- ZA-FINAL6 (Round T〜Z2): 富の凝縮の根治 ----
# 診断の系譜 (「最近ほど大変動」「暴騰暴落の頻度が実の3-4倍」の真因):
#   平穏ボラ床の経年ラチェット (0.71→1.02、実は横ばい0.6) を順に切り分け:
#   realized_varフィードバック(T:アンカー無効)→asym_pi(自己正規化済)→ストレス(平穏0)
#   →現金/保有乖離(比率安定)→score(横ばい)→rebalance(無効)→trend_weight(R:不変)
#   → **富の凝縮**: 60年で実効投資家数 14.8→1人 (HHI 0.995)、金額imbalance飽和
#     0.39→0.91 が内生ボラ床を押し上げていた。
# 対処の試行:
#   U(即時交代)/V(段階的清算付き資本回転): 99%ホエールの退出はどの方式でも市場地震 → 棄却
#   W: **キャパシティ制約** (大口の執行制約) γ=0.5 — 注文×(富シェア×n)^(-γ)。
#      床ラチェット完全解消 (1.46→0.95)、バブル頻度 1.7%→0.3%。
#   X: γで大人しくなった分ベースノイズ復元 (exog/mvol を FINAL4 水準へ)
#   Y: drift 0.00030 (成長 0.074=実)。asym_pi強化は飽和ぎみ (2.4採用)
#   Z1: 売りγ常時解放は leverage 完全復元だがラチェット復活 → トレードオフ
#   Z2: **状態依存ファイアセール** — 下方ストレス比 down_var/(realized/2) > 0.8 で
#       売りγを解除。平時ラチェット抑制と危機増幅 (leverage) を両立。
# 結果 (3seed): worst21 -0.357(実-0.356) kurt 20.9(実21.8) max12m 0.80(実0.77)
#   床 1.06 freq 0.011(実0.006) std 0.012 lev_dec10_10d -0.166(実-0.176)
#   lev_dec10_5d -0.172(実-0.209、8割回復) ann 0.074(実0.0745)
# 資本回転 (turnover) 機構はコードに残るが FINAL6 では無効 (0.0)。
ZA_FINAL6_PARAMS = dict(
    ZA_FINAL5_PARAMS,
    whale_size_power=0.5,
    exog_common_sigma=0.0064,
    market_vol=0.0045,
    market_anchor_drift=0.00030,
    asym_pi_scale=2.4,
    whale_fire_sale_relief=1.0,
    whale_fire_thresh=0.8,
)


# ---- ZA-FINAL7 (Round AA): 窓内ボラクラスタリングの回復 ----
# 5年窓分布の診断 (実12チャンク vs 生成200窓): abs_acf5 実0.205 vs 0.081、
# sq_acf5 実0.163 vs 0.067 — Q〜Zでiid外生ノイズ比重が増え窓内の持続が痩せた。
# 遅い余震 (半減期46日) jump_aftershock2_scale 6→10 で ACF全域を改善:
#   abs1 0.257(実0.261) abs5 0.199(実0.289) abs20 0.087(実0.191)
#   sq5 0.145(実0.228) sq20 0.051(実0.067)  aggregate score 0.302 (最良)
# トレードオフ (ユーザー判断で採用): worst21 -0.394(実-0.356) q999 0.087(実0.070)
#   lev_dec10 -0.150(実-0.209) freq 0.014(実0.006)。
# AB (disaster_sigma縮小で相殺) は逆効果で棄却。
# 残オープン: lag5-20 のボラクラは実の~70% (多時間スケールカスケードの欠如、
# ZA-LEAN-LM 期からの構造的限界)。
ZA_FINAL7_PARAMS = dict(
    ZA_FINAL6_PARAMS,
    jump_aftershock2_scale=10.0,
)
