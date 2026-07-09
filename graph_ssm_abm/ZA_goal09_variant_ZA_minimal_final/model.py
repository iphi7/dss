"""
P_base5_variant_P_leverage/model.py

O19 をベースに、leverage effect（下落後にボラが上がる非対称性）を出すための
複数機構を検証する。各機構は独立した Config スイッチで有効/無効を切り替えられる。

追加機構:
  1. GJR-GARCH 的 c_t 分散増幅 (gjr_scale > 0)
       下落後に c_t の sigma を一時的に上昇させる。
       down_var_ewma = decay * down_var_{t-1} + (1-decay) * max(-r_SP_t, 0)^2
       sigma_c_t = sigma_c * sqrt(1 + gjr_scale * down_var_ewma / sigma_c^2)

  2. 非対称価格インパクト (asym_pi_scale > 0)
       下落後に lambda_t（price impact 係数）を増幅する。
       lambda_t *= 1 + asym_pi_scale * down_var_ewma / market_vol^2

  3. 損失トリガー強制売り / stop-loss (stoploss_scale > 0)
       市場下落時（前期 sp_ret < 0）に、リスク回避型投資家（vol_sensitivity < 0）の
       売りオッズを増幅する。損失を被った投資家が追加で売りを出す行動を模倣。

  4. 非対称クラッシュ (asym_crash_sell_only = True)
       流動性クラッシュを売り超過時のみ発動させる。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    seed: int = 42
    n_days: int = 1260
    n_firms: int = 80
    n_investors: int = 60
    ba_m: int = 3
    n_sectors: int = 8

    # ===== 投資ユニバース制限 (ZA系 スケール不変性) =====
    # 'all' = 全銘柄 (従来)。'sector_graph' = 専門セクター全社 + グラフ隣接 + ランダム少数。
    # 不均衡の粒度 sd(I_j) ~ sqrt(H_j) は銘柄 j の参加者数で決まるため、全員全銘柄では
    # 投資家数 N の増加で取引由来ボラが死ぬ (F02)。ユニバースサイズ k を N によらず一定に
    # 保てば企業あたり参加者数 N·k/n が (N/n 固定のもとで) 一定になり、粒度が保存される。
    # スケール変更時は sector size ≈ n/n_sectors を固定して n_sectors を n に比例させること。
    universe_mode: str = "all"
    universe_random_extra: int = 5
    # ユニバースサイズの上限 (0=無制限)。専門セクター銘柄は必ず保持し、
    # グラフ隣接+ランダム分を超過分だけ間引く。スケールとともにハブ隣接が
    # 膨らんで参加者密度が上がるのを防ぎ、銘柄あたり参加者数を一定に保つ。
    universe_max_size: int = 0
    # 銘柄側の参加者数キャップ (0=無制限)。BAハブは全セクターの隣接集合に入るため
    # スケールとともにハブの参加者数だけが増え、leverage が大スケールで希釈される。
    # 専門セクターとしての参加は常に許可し、隣接/ランダム経由の追加参加を制限する。
    universe_hub_cap: int = 0

    n_pub:  int = 6
    n_sec:  int = 8
    n_priv: int = 6

    phi_dims: Tuple[float, ...] = (
        0.88, 0.86, 0.84, 0.83, 0.82, 0.80,
        0.75, 0.73, 0.72, 0.70, 0.68, 0.67, 0.65, 0.63,
        0.61, 0.59, 0.57, 0.55, 0.54, 0.52,
    )
    rho_dims: Tuple[float, ...] = (
        0.06, 0.07, 0.08, 0.09, 0.10, 0.12,
        0.17, 0.18, 0.20, 0.21, 0.22, 0.23, 0.25, 0.27,
        0.28, 0.29, 0.30, 0.32, 0.33, 0.35,
    )
    eta_dims: Tuple[float, ...] = (
        0.015, 0.016, 0.016, 0.017, 0.018, 0.018,
        0.020, 0.021, 0.022, 0.023, 0.023, 0.024, 0.025, 0.026,
        0.014, 0.014, 0.013, 0.013, 0.012, 0.012,
    )

    obs_sigma_pub: float = 0.040
    obs_sigma_sec: float = 0.050

    dim_weights: Tuple[float, ...] = (
        0.080, 0.080, 0.080, 0.080, 0.080, 0.080,
        0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040,
        0.033, 0.033, 0.034, 0.033, 0.033, 0.034,
    )

    mktcap_pareto_a:     float = 1.2
    mktcap_degree_power: float = 0.7
    mktcap_degree_reference: str = "true"

    rare_shock_prob:  float = 0.018
    rare_shock_sigma: float = 0.12

    price_impact:       float = 0.050
    idio_vol:           float = 0.0075
    market_vol:         float = 0.0060
    common_shock_beta:  float = 1.00
    market_garch_alpha: float = 0.050
    market_garch_beta:  float = 0.880
    leverage_vol_beta:  float = 0.010
    vol_persistence:    float = 0.94

    exog_common_sigma:      float = 0.0040
    exog_common_jump_prob:  float = 0.006
    exog_common_jump_sigma: float = 0.035
    exog_common_clip:       float = 0.100
    # 日常共通ノイズの状態依存化 (0=無効)。固定 sigma は静穏期のフロアになり、
    # 実データの「究極に静かな時期」(rolling std 0.006) やリターン分布の
    # 中心質量 (|r|<0.2% が23%) を再現できない。実現ボラへの緩いべき連動で
    # 静穏期に sigma_c を下げる: sigma_c,t = sigma_c × clip((実現ボラ/2σ0)^c, 0.6, 1.5)
    exog_sigma_vol_coupling: float = 0.0
    realized_vol_lambda:    float = 0.985

    garch_stress_scale: float = 0.0
    garch_down_scale:   float = 0.3

    vol_sensitivity_mean: float = 0.80
    vol_sensitivity_std:  float = 0.80

    wealth_sigma:          float = 1.20
    wealth_vol_corr:       float = 1.20
    wealth_clip_min:       float = 0.10
    wealth_clip_max:       float = 12.0

    participation_vol_power: float = 1.80

    impact_activity_scale: float = 2.50
    impact_activity_clip:  float = 6.00
    impact_crash_threshold: float = 1.20
    impact_crash_scale:     float = 4.00
    impact_crash_power:     float = 2.00

    vol_activity_scale:        float = 0.0
    vol_activity_ewma_lambda:  float = 0.94

    initial_sp500_abs: float | None = None
    initial_dgs10_abs: float | None = None
    exog_drift: float = 0.0   # 日次リターンへの外生ドリフト (長期価格安定化用)
    use_graph: bool = True

    graph_topology: str = "ba"
    subjective_graph_mode: str = "partial"

    alpha_max: float = 0.85
    alpha_pub_base: float = 0.45
    alpha_pub_nonexpert_scale: float = 0.70
    alpha_sec_expert: float = 0.60
    alpha_sec_nonexpert: float = 0.05
    alpha_priv_base: float = 0.25
    alpha_priv_nonexpert_scale: float = 0.30
    update_style_sigma: float = 0.25

    private_anchor_scale: float = 1.00
    rho_h_dims: Tuple[float, ...] = (0.90, 1.00, 1.20, 1.00, 0.90, 1.10)

    # 投資家の銘柄選択スコアを「絶対評価」から「相対評価」へ寄せる係数。
    # 1.0なら各投資家内で平均スコアを差し引き、全銘柄への共通な売り/買いバイアスを弱める。
    # uncertainty や金利ペナルティが全銘柄に同じ符号で乗ると長期で市場全体が沈みやすいため、
    # 60年生成ではここを少し有効化する。
    score_centering: float = 0.0
    market_risk_premium_score: float = 0.0
    # raw return ACF 抑制: 前期リターン momentum 項の係数。P/Qでは固定 0.25 だった。
    momentum_score_weight: float = 0.25
    # ACF/ボラ過剰抑制: 価格形成で使う firm return の日次クリップ。
    firm_return_clip: float = 0.18
    # 60年生成用: 投資家の保有/現金が極端な長期レジームを作らないようにする弱いturnover。
    # 各日、ポートフォリオ価値の一部を市場ウェイトのターゲットへ寄せる。
    portfolio_rebalance_rate: float = 0.0
    portfolio_cash_target: float = 0.05

    # ===== 投資家ヘテロ性による中期ストレス反応 (Y系追加) =====
    # 下落後の市場ストレスを、全員一律のfearではなく投資家タイプ別の参加率・注文量へ写す。
    investor_stress_scale: float = 0.0
    investor_stress_decay: float = 0.94
    investor_stress_threshold: float = 0.0
    investor_stress_clip: float = 4.0
    risk_pref_participation_scale: float = 0.0
    risk_pref_size_scale: float = 0.0
    risk_averse_withdraw_scale: float = 0.0
    risk_pref_buy_tilt: float = 0.0
    risk_pref_sell_tilt: float = 0.0

    # ===== Z系追加: ジャンプ tail 強化・余震・中位分位 leverage =====
    # 共通ジャンプの t 分布自由度。小さいほど裾が重くなる (従来は 3 固定)。
    jump_df: float = 3.0
    # 余震 (aftershock): 大ジャンプ後の数日間、共通ノイズの分散を一時的に増幅する。
    # 0 で無効。ジャンプ絶対値を状態に加算し、日次で指数減衰させる。
    jump_aftershock_scale: float = 0.0
    jump_aftershock_decay: float = 0.60
    # ジャンプサイズのボラ連動。0 で無効。>0 のとき、ジャンプ振幅に
    # (実現ボラ / 基準ボラ)^coupling を乗じる (係数は [0.3, 3.0] にクランプ)。
    # 静穏期の不自然な巨大ジャンプ (尖度爆発の源) を抑え、活発期の tail を保つ。
    jump_vol_coupling: float = 0.0
    # 2層ジャンプの第2層 (中規模・やや高頻度)。実データの Q-Q は
    # 「そこそこの頻度の中規模ジャンプ + ごく稀な特大ジャンプ」の2層構造を示す。
    # 第1層 (exog_common_jump_*) を特大・超レアに、第2層を中規模・年数回に使う。
    # 第2層も余震状態に加算されるため、sq_acf のクラスタ源になる。
    exog_common_jump2_prob: float = 0.0
    exog_common_jump2_sigma: float = 0.0
    jump2_df: float = 5.0
    # メガクラッシュ: 期間内1回程度の単日 -20% 級イベント (1987型)。
    # 実データのジャンプは「~0.2 が1回 + 0.075~0.1 がそこそこ」の2階層に
    # くっきり分かれるため、特大側を固定サイズの独立層として持つ。
    # サイズは mega_crash_size × Uniform(0.9, 1.1)、符号は負 (クラッシュ)。
    # 事後は余震状態に加算される (1987後の +5%/+9% 日に対応)。
    mega_crash_prob: float = 0.0
    mega_crash_size: float = 0.20
    # メガクラッシュが災害エピソードを誘発する強度 (0 = 誘発しない)。
    # 1987 型の「暴落 → 数日間の ±5〜9% 乱高下」を再現し、孤立した単日巨大 r^2 が
    # 二乗リターン ACF を希釈する問題を防ぐ。
    mega_triggers_episode: float = 0.0
    # 遅い第2余震成分。第1余震 (数日) より長い 2〜3週間スケールの分散増幅で、
    # 二乗リターン ACF の lag10-20 の持続源になる。
    jump_aftershock2_scale: float = 0.0
    jump_aftershock2_decay: float = 0.96
    # 災害エピソード (複数日クラッシュ)。単日の特大ジャンプの構造的置き換え。
    # 実データの危機 (2008, 2020) は -5〜-9% の日が数日連続する形で、
    # 単日 -20% は例外 (1987)。トリガー日に強度 1.0 で開始し、毎日
    # t分布ドロー × disaster_sigma × 強度 を共通ノイズに加算、強度は日次で減衰。
    # 連続する大きな |r| が r^2 クラスタを作り、sq_acf の構造的な源になる。
    disaster_prob: float = 0.0        # 1日あたりのエピソード開始確率
    disaster_sigma: float = 0.05      # エピソード日次ショックのスケール
    disaster_df: float = 4.0          # 日次ショックの t 分布自由度
    disaster_decay: float = 0.65      # 強度の日次減衰率
    disaster_end: float = 0.10        # 強度がこの値未満になるとエピソード終了
    # プラトー日数: 強度が 1.0 のまま維持される日数 (その後幾何減衰)。
    # 実データの sq_acf は lag5 まで平坦で、危機ボラが数日でなく1〜2週間
    # 高止まりすることを示す。幾何減衰のみでは lag5 で相関が切れる。
    disaster_plateau: int = 0
    # エピソード日次ショックの平均シフト (強度1あたり)。負にすると危機週が
    # net 下落になり (現実の危機と整合)、大きな下落日→高ボラ持続の連鎖が
    # 内生的な dec10 leverage を作る (外生対称ショックは leverage を希釈する)。
    disaster_mu: float = 0.0
    # 下落記憶のデッドバンド。この閾値未満の下落は down_var_ewma・投資家ストレスに
    # 蓄積しない。微小な下落だけが記憶に入る非対称性が低分位 (3-4) の
    # 正の leverage 相関を作るアーティファクトへの構造的対応。
    down_deadband: float = 0.0
    # 相対デッドバンド: 実現ボラ × この係数を閾値に使う。絶対デッドバンドは
    # 静穏レジームで通常サイズの下落まで遮断し発火を殺すため (Round11 Z44で確認)、
    # 現在のボラ水準に適応する相対形が正しい。
    # → Round12 で相対形も発火と両立しないことを確認。非推奨。
    down_deadband_rel: float = 0.0
    # 下落記憶の閾値なし線形成分 (0=無効)。leverage機構 (asym_pi/ストレス) は
    # neg² と閾値で「意味のある下落」だけに反応するため、低中ボラ分位の
    # 小さな下落→将来ボラの滑らかな応答の裾が構造的にゼロになる。
    # down_var とストレスの蓄積に + coef×neg×market_vol の線形項を併設する
    # (小さい neg では neg² に匹敵、大きい neg では無視できる)。
    down_linear_coef: float = 0.0
    # 公開観測の momentum 項のスケール (1.0 = 従来)。低分位 leverage の正相関の
    # 原因切り分け用: 正リターン → momentum で観測強気 → 買い活動増 → ボラ増、
    # という上昇側チャネルを弱められる。
    obs_momentum_scale: float = 1.0
    # 非対称価格インパクトの活性化閾値。1.0 が従来 (baseline 超過分のみ)。
    # 1.0 未満にすると中程度の下落でも活性化し、中位分位の leverage を作る。
    asym_pi_threshold: float = 1.0
    # 非対称価格インパクトの飽和 (tanh)。0 で無効 (線形)。
    # 有効時は過大な超過が飽和し、最上位分位の leverage 過剰を抑える。
    asym_pi_sat: float = 0.0
    # 投資家別ストレス記憶長。decay_max > decay_min > 0 のとき有効。
    # リスク選好度 × 資産の大きい投資家ほど記憶が長い (decay が大きい)。
    investor_stress_decay_min: float = 0.0
    investor_stress_decay_max: float = 0.0
    # 参加率の下限。低ボラ時に取引が消滅する自己強化トラップ
    # (低ボラ→参加減→取引由来ボラ消失→低ボラ) を防ぐ。
    participation_floor: float = 0.10
    # ボラ感応係数と参加率の上限。高ボラ状態の自己維持 (高ボラ→高参加→高ボラ) の
    # ゲインを制御する。下げると高温レジームが準安定になり、消火遷移が起きやすくなる。
    vol_factor_max: float = 4.0
    participation_cap: float = 5.0
    # 参加率への日次ノイズ (対数正規 σ)。平常時のボラ→参加→ボラの滑らかな持続が
    # |r| の lag1 自己相関を過剰にするため、日々の参加に iid 揺らぎを入れて希釈する。
    participation_noise_sigma: float = 0.0

    # ===== DGS10 とのレジーム依存相関 (Z系 Round24+) =====
    # 実データの90日ローリング相関 corr(株リターン, 金利変化) は
    # 高金利期 (1966-97) に -0.2〜-0.4、低金利期 (2000-19) に +0.2〜+0.5、
    # 金利4%台の現在はゼロ近傍と、金利水準に依存して符号が切り替わる。
    # レジーム係数: kappa_t = tanh((rate_regime_center - DGS10水準) / rate_regime_width)
    #   高金利 → kappa<0 (割引率チャネル: 金利上昇で株安)
    #   低金利 → kappa>0 (リスクオン: 良いニュースで株高&金利上昇)
    rate_regime_center: float = 4.5   # % 単位。0 以下で本機構無効
    rate_regime_width: float = 1.5    # 遷移の滑らかさ (%)
    # 経路A: 価格直接 — firm_return += rate_price_beta * kappa_t * (DGS10日次変化)
    rate_price_beta: float = 0.0
    # 経路B: 投資家スコア経由 — score += beta * kappa_t * (DGS10日次変化) * (感応度_i/0.10)
    rate_change_score_beta: float = 0.0
    # 金利上昇トレンド補正: 実効水準 = 水準 + scale × max(1年前比の上昇, 0)。
    # 2020s は水準が低い(4%台)のに相関ゼロなのは金利急上昇(インフレ)で
    # 割引率チャネルが復活したため。1966-69 (低水準・上昇中・負相関) とも整合。
    rate_trend_scale: float = 0.0
    rate_trend_window: int = 250

    # ===== DGS10 の内生生成 (Z系 Round27+) =====
    # True にすると DGS10 を実データ入力でなくモデルで生成する。
    # 実データの特性: 水準はほぼ単位根 (lag250自己相関0.92) + 数十年スケールの大波
    # (4.6→15.8→0.5→4.5)、日次変化 std=0.067pp・尖度8.4、|変化|自己相関0.25 (ボラクラ)、
    # ボラは水準の弱いべき乗 (gamma≈0.32)。
    generate_dgs10: bool = False
    dgs10_init: float = 4.63          # 初期水準 (%; 実データ1966年初と同じ既定)
    dgs10_sigma0: float = 0.052       # 水準5%時の日次変化の基準スケール (pp)
    dgs10_vol_gamma: float = 0.35     # ボラの水準べき指数
    dgs10_df: float = 5.0             # 日次変化の t 分布自由度 (尖度源)
    dgs10_vol_lambda: float = 0.95    # 分散状態 EWMA (|変化|クラスタの源)
    # 遅い第2分散成分 (0=無効)。実データの |Δy| は lag60 まで平坦な長期記憶を持ち、
    # 単一の速い EWMA では再現できない。月〜年スケールの第2成分を乗算合成する。
    dgs10_vol_lambda2: float = 0.0
    # 持続的ドリフト = インフレレジーム。AR(1) で数年〜十年単位の上昇/下降期を作る
    dgs10_drift_rho: float = 0.9995
    dgs10_drift_sigma: float = 0.00008
    # 弱い平均回帰と反射境界 (60年で発散/貼り付きしないように)
    dgs10_mr_theta: float = 0.00002
    dgs10_mr_center: float = 5.5
    dgs10_min: float = 0.4
    dgs10_max: float = 16.0
    # 株→金利チャネル: 当日の株リターン × kappa を金利変化に加算 (pp / 単位リターン)。
    # リスクオン期 (kappa>0) は株安→金利低下 (質への逃避)、高金利期は割引率チャネルの逆向き。
    # 生成モードでは同日相関の主たる源 (株の取引後に当日の金利変化を生成する)。
    dgs10_stock_beta: float = 0.0
    # 極端日の flight-to-quality/relief 結合 (レジーム kappa に依存しない常に正の結合)。
    # 実データでは |r| の極端分位のみ corr(r, Δy) が正に跳ね (急落→国債買い→金利低下、
    # 1987 は高金利時代でも金利急低下)、中間分位はレジーム混合の弱い負。
    # Δy += flight_beta × sign(r) × max(|r| − flight_thresh × 実現ボラ, 0)
    dgs10_flight_beta: float = 0.0
    dgs10_flight_thresh: float = 1.2
    # flight の持続状態 (>0 で状態変数型)。|r| が閾値を超えると状態=1 に点火し
    # 日次で減衰。状態が生きている間は当日リターン全体が正結合する
    # (実データの flight は危機週全体で株安→金利低下が続くエピソード現象。
    #  単日の超過分だけの結合では7日窓相関の極端分位に現れない)。
    dgs10_flight_decay: float = 0.0
    # レジーム符号付き増幅 (0=無効)。極端日 (|r| > amp_thresh×実現ボラ) で点火し
    # amp_decay で減衰する状態の間、株→金利結合を (1 + amp×state) 倍にする。
    # 旧 flight (常に正) の修正版: 実データは時代内で「高ボラほどその時代の
    # レジーム相関が増幅」(負時代 d20=-0.39、正時代 d20=+0.55) するため、
    # kappa の符号を保ったまま結合強度だけ増幅するのが正しい構造。
    dgs10_regime_amp: float = 0.0
    dgs10_amp_thresh: float = 2.0
    dgs10_amp_decay: float = 0.80
    # 危機エピソード限定の flight (0=無効)。disaster_intensity > 0 の間だけ
    # 当日リターンに正結合 (レジームによらず暴落週は株安→金利低下)。
    # 1987 (高金利時代の暴落でも金利急落) に対応。通常の高ボラ日は
    # レジーム相関のままなので、時代内の負の増幅と両立する。
    dgs10_episode_flight: float = 0.0
    # ===== リードラグ (ZA系 フェーズ3) =====
    # 株トレンド→金利ドリフト (景気期待): 株の20日平均リターンを金利が遅れて追う。
    # 実データの「株先行の正相関 (+20〜45日ラグで+0.10)」に対応。
    dgs10_stocktrend_beta: float = 0.0
    # 金利トレンド→株スコア (割引率の遅効): 過去20日の外生金利変化の平均が
    # 株スコアを負に圧迫する。「金利先行の負相関 (-7日ラグで-0.15)」に対応。
    # 入力は外生成分のみ (株→金利→スコア→株のループ防止)。
    rate_lag_score_beta: float = 0.0
    # 正側 kappa の非対称スケール (1.0 = 対称)。tanh は低金利側で速く飽和するため
    # 時間加重 kappa が正に偏り、中間分位の相関が正に浮く。正側だけ縮めて補正する。
    rate_kappa_pos_scale: float = 1.0
    # 金利変化の日次クリップ (実データ max |Δ|=0.75pp)
    dgs10_change_clip: float = 0.9
    # True のとき、投資家スコアの金利変化項は「外生成分」(Δy から株由来項
    # β_sr×kappa×r を除いた部分) に反応する。生成モードでは
    # 株→金利→翌日スコア→株 の1日往復が生リターンの正の自己相関を合成するため、
    # このループを切る (昨日の自分たちの売買が動かした金利には二重反応しない)。
    rate_score_use_exo: bool = False
    # 投資家母集団の層化抽出。True にすると wealth と vol_sensitivity を
    # 分位数グリッドから生成し (ペアリングのみ乱数)、母集団の抽出誤差を除去する。
    # iid 抽出では n=60 程度だと「大口リスク選好投資家 (クジラ)」の有無が seed 次第となり、
    # 市場のレジーム構造 (発火するか否か) が母集団ガチャで決まってしまう。
    investor_stratified: bool = False

    # ===== Leverage 機構 (P 系追加) =====
    # 1. GJR-GARCH: 下落後に c_t sigma を増幅
    gjr_scale: float = 0.0        # > 0 で有効
    gjr_centered: bool = False    # True = ratio to realized_var (平常時に影響なし)

    # 2. 非対称価格インパクト: 下落後に lambda_t を増幅
    asym_pi_scale: float = 0.0    # > 0 で有効
    asym_pi_centered: bool = False  # True = ratio to realized_var (平常時に影響なし)

    # 共通 EWMA 減衰率 (gjr / asym_pi 両方で使用)
    down_ewma_decay: float = 0.95

    # 3a. stop-loss (リスク回避型のみ): 前期下落時に売りオッズ増幅
    stoploss_scale: float = 0.0   # > 0 で有効 (vol_sensitivity < 0 の投資家のみ)
    # 3b. market-wide fear: 下落後に全投資家の売りオッズを増幅
    stoploss_universal_scale: float = 0.0     # > 0 で有効 (全投資家)
    stoploss_universal_threshold: float = 0.005  # 発動する最小下落幅 (例: 0.5%)

    # 4. 非対称クラッシュ: 売り超過時のみ発動
    asym_crash_sell_only: bool = False

    # ===== 長期安定化: ファンダメンタル価値アンカー (Q系) =====
    # 企業価格が基準価値から大きく乖離したときだけ、弱い平均回帰リターンを追加する。
    fundamental_strength: float = 0.0      # κ。0で無効
    fundamental_gap_scale: float = 0.50    # tanh のスケール。log gap がこの程度で飽和
    fundamental_clip: float = 0.006        # 1日あたりの最大アンカー寄与
    fundamental_drift: float = 0.00025     # 基準価値の平均日次成長
    fundamental_noise: float = 0.0008      # 基準価値の企業別日次ノイズ
    fundamental_state_sensitivity: float = 0.0000  # 企業潜在状態から基準価値成長への寄与

    # 市場全体の長期ファンダメンタル指数アンカー。
    # 個別企業ではなくSP500水準そのものの崩壊/爆発を抑える弱い復元力。
    market_anchor_strength: float = 0.0
    # ZA系: アンカー成長率のパスごと確率ドリフト (0=無効)。
    # 決定論的アンカーは全seedの終値を±15%に束ねる。持続AR(1)のドリフト変動で
    # パスごとの長期軌道の多様性を作る (金利のインフレレジームと同じ構造)。
    market_anchor_drift_sigma: float = 0.0
    market_anchor_drift_rho: float = 0.9995
    market_anchor_gap_scale: float = 0.70
    market_anchor_clip: float = 0.004
    market_anchor_drift: float = 0.00025


def make_ba_graph(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    if m < 1 or m >= n:
        raise ValueError("ba_m must satisfy 1 <= m < n_firms")
    adj = np.zeros((n, n), dtype=float)
    degrees = np.zeros(n, dtype=float)
    start = m + 1
    for u in range(start):
        for v in range(u + 1, start):
            adj[u, v] = adj[v, u] = 1.0
            degrees[u] += 1
            degrees[v] += 1
    for new in range(start, n):
        probs = degrees[:new] / degrees[:new].sum()
        targets = rng.choice(new, size=m, replace=False, p=probs)
        for target in targets:
            adj[new, target] = adj[target, new] = 1.0
            degrees[new] += 1
            degrees[target] += 1
    weights = rng.uniform(0.25, 1.0, size=(n, n))
    adj *= (weights + weights.T) / 2.0
    return row_normalize(adj)


def row_normalize(mat: np.ndarray) -> np.ndarray:
    row_sum = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)


def make_subjective_graphs(
    true_w: np.ndarray,
    sectors: np.ndarray,
    n_investors: int,
    rng: np.random.Generator,
    vol_sensitivity_mean: float = 0.80,
    vol_sensitivity_std:  float = 0.80,
) -> tuple[np.ndarray, pd.DataFrame]:
    n = true_w.shape[0]
    graphs = np.zeros((n_investors, n, n), dtype=float)
    rows = []

    for i in range(n_investors):
        expertise = int(rng.integers(0, sectors.max() + 1))
        base_keep = rng.uniform(0.15, 0.55)
        expert_bonus = rng.uniform(0.20, 0.40)
        false_edge_prob = rng.uniform(0.002, 0.015)

        keep_prob = np.full((n, n), base_keep)
        expert_mask = (sectors[:, None] == expertise) | (sectors[None, :] == expertise)
        keep_prob = np.where(expert_mask, np.minimum(0.95, keep_prob + expert_bonus), keep_prob)
        keep = rng.random((n, n)) < keep_prob
        subjective = np.where(keep, true_w, 0.0)
        false_edges = (rng.random((n, n)) < false_edge_prob) & (true_w == 0)
        subjective = np.where(false_edges, rng.uniform(0.02, 0.15, size=(n, n)), subjective)
        np.fill_diagonal(subjective, 0.0)
        subjective = row_normalize(subjective)
        graphs[i] = subjective

        vol_sens = float(np.clip(rng.normal(vol_sensitivity_mean, vol_sensitivity_std), -2.0, 3.0))

        rows.append({
            "investor_id": i,
            "expertise_sector": expertise,
            "edge_keep_base": base_keep,
            "risk_tolerance": rng.lognormal(mean=-2.6, sigma=0.35),
            "vol_sensitivity": vol_sens,
            "trend_weight": rng.normal(0.28, 0.12),
            "value_weight": rng.normal(1.00, 0.25),
            "uncertainty_aversion": rng.uniform(0.15, 0.75),
            "rate_sensitivity": rng.uniform(0.02, 0.18),
            "temperature": rng.uniform(2.0, 6.0),
            "loss_asymmetry": rng.uniform(1.0, 1.9),
            "belief_phi": float(np.clip(rng.normal(0.72, 0.07), 0.50, 0.90)),
            "belief_rho_s": float(np.clip(rng.normal(0.18, 0.06), 0.04, 0.30)),
            "belief_rho_h": float(np.clip(rng.normal(0.28, 0.08), 0.10, 0.52)),
            "obs_var": rng.lognormal(mean=np.log(0.055**2), sigma=0.45),
            "proc_var": rng.lognormal(mean=np.log(0.022**2), sigma=0.45),
            "recognized_edges": int((subjective > 0).sum()),
        })

    return graphs, pd.DataFrame(rows)


def simulate_market(
    output_df: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(config.seed)
    n = config.n_firms
    t_max = config.n_days
    n_pub  = config.n_pub
    n_sec  = config.n_sec
    n_priv = config.n_priv
    d = n_pub + n_sec + n_priv

    if len(output_df) < t_max + 2:
        raise ValueError("output.csv is shorter than requested n_days")

    historical_tail = output_df.tail(t_max).reset_index(drop=True)
    last_row = output_df.iloc[-1]
    initial_sp    = float(config.initial_sp500_abs or last_row["sp500_abs"])
    initial_dgs10 = float(config.initial_dgs10_abs or historical_tail.iloc[0]["DGS10_abs"])

    dates = pd.bdate_range(
        pd.to_datetime(last_row["Date"]) + pd.offsets.BDay(1), periods=t_max
    )

    ba_reference_w = make_ba_graph(n, config.ba_m, rng)
    if config.graph_topology == "zero":
        true_w = np.zeros((n, n), dtype=float)
    elif config.graph_topology == "ba":
        true_w = ba_reference_w.copy()
    else:
        raise ValueError(f"unknown graph_topology: {config.graph_topology}")

    sectors = rng.integers(0, config.n_sectors, size=n)
    subjective_graphs, investor_df = make_subjective_graphs(
        true_w, sectors, config.n_investors, rng,
        vol_sensitivity_mean=config.vol_sensitivity_mean,
        vol_sensitivity_std=config.vol_sensitivity_std,
    )
    if config.graph_topology == "zero":
        subjective_graphs[:] = 0.0
        investor_df["recognized_edges"] = 0
    elif config.subjective_graph_mode == "perfect":
        subjective_graphs[:] = true_w[None, :, :]
        investor_df["recognized_edges"] = int((true_w > 0).sum())
    elif config.subjective_graph_mode != "partial":
        raise ValueError(f"unknown subjective_graph_mode: {config.subjective_graph_mode}")

    if np.mean(np.abs(true_w)) > 0:
        graph_quality = []
        denom_w = np.mean(np.abs(true_w)) + 1e-12
        for i in range(config.n_investors):
            q = 1.0 - np.mean(np.abs(subjective_graphs[i] - true_w)) / denom_w
            graph_quality.append(float(np.clip(q, 0.0, 1.0)))
        investor_df["graph_quality"] = graph_quality
    else:
        investor_df["graph_quality"] = 0.0

    investor_df["update_style"] = np.clip(
        rng.lognormal(mean=0.0, sigma=config.update_style_sigma, size=config.n_investors),
        0.5, 1.5,
    )

    vol_s = investor_df["vol_sensitivity"].to_numpy(dtype=float)
    vol_s_z = (vol_s - vol_s.mean()) / (vol_s.std() + 1e-12)
    wealth_log = rng.normal(0.0, config.wealth_sigma, size=config.n_investors)
    wealth_log += config.wealth_vol_corr * vol_s_z
    wealth_factor = np.exp(wealth_log)
    wealth_factor = np.clip(wealth_factor, config.wealth_clip_min, config.wealth_clip_max)
    wealth_factor = wealth_factor / wealth_factor.mean()
    investor_df["wealth_factor"] = wealth_factor

    if config.mktcap_degree_reference == "true":
        degree_for_mktcap = (true_w > 0).sum(axis=1).astype(float)
    elif config.mktcap_degree_reference == "ba_reference":
        degree_for_mktcap = (ba_reference_w > 0).sum(axis=1).astype(float)
    else:
        raise ValueError(f"unknown mktcap_degree_reference: {config.mktcap_degree_reference}")

    base_shares = rng.pareto(a=config.mktcap_pareto_a, size=n) + 1.0
    if degree_for_mktcap.mean() > 0:
        degree_factor = (degree_for_mktcap / degree_for_mktcap.mean()) ** config.mktcap_degree_power
    else:
        degree_factor = np.ones(n, dtype=float)
    shares = base_shares * degree_factor
    shares = shares / shares.mean()

    firm_prices = np.full(n, initial_sp) * rng.lognormal(mean=0.0, sigma=0.12, size=n)
    market_caps = firm_prices * shares

    phi   = np.array(config.phi_dims)
    rho   = np.array(config.rho_dims)
    eta   = np.array(config.eta_dims)
    dim_w = np.array(config.dim_weights)

    x      = rng.normal(0.0, 0.020, size=(n, d))
    x_prev = x.copy()

    cash     = wealth_factor.astype(float).copy()
    holdings = rng.lognormal(mean=-5.0, sigma=0.4, size=(config.n_investors, n)) * wealth_factor[:, None]
    belief_state = rng.normal(0.0, 0.010, size=(config.n_investors, n, d))
    rho_h_dim = np.array(config.rho_h_dims, dtype=float)

    firm_vol          = np.full(n, config.idio_vol)
    market_var        = config.market_vol ** 2
    prev_common_noise = 0.0
    prev_market_vol_t = config.market_vol
    realized_var_ewma = config.market_vol ** 2
    vol_trade_ewma_fast  = 0.0
    vol_trade_ewma_slow  = 0.0
    vol_trade_ewma_initialized = False
    sp_abs = initial_sp

    # Leverage 機構用の状態変数
    # down_var_ewma: 下落リターンの EWMA 二乗平均 (GJR / 非対称 PI 共用)
    down_var_ewma = config.exog_common_sigma ** 2
    prev_sp_ret = 0.0  # stop-loss 判定用

    records      = []
    firm_records = []

    params = investor_df.to_dict("records")

    for t in range(t_max):
        dgs10_abs    = float(historical_tail.loc[t, "DGS10_abs"])
        dgs10_change = float(historical_tail.loc[t, "DGS10"])
        rf_level     = dgs10_abs / 100.0

        rare = rng.random(n) < config.rare_shock_prob
        for k in range(d):
            noise_k = rng.normal(0.0, eta[k], size=n)
            if k == 0:
                noise_k += rare * rng.standard_t(df=3, size=n) * config.rare_shock_sigma
            x[:, k] = phi[k] * x[:, k] + rho[k] * (true_w @ x[:, k]) + noise_k

        y_pub = np.empty((n, n_pub))
        for k in range(n_pub):
            mom = 0.35 if k == 0 else 0.15
            y_pub[:, k] = x[:, k] + mom * (x[:, k] - x_prev[:, k]) + rng.normal(0.0, config.obs_sigma_pub, size=n)

        y_sec = np.empty((n, n_sec))
        for k_loc in range(n_sec):
            k_glob = n_pub + k_loc
            y_sec[:, k_loc] = x[:, k_glob] + rng.normal(0.0, config.obs_sigma_sec, size=n)

        y_obs_global = np.concatenate([y_pub, y_sec], axis=1)
        anchor_specs = [
            [(0, 0.60), (3, 0.40)],
            [(1, 0.50), (6, 0.50)],
            [(8, 0.50), (9, 0.50)],
            [(4, 0.50), (10, 0.50)],
            [(2, 0.50), (3, 0.50)],
            [(11, 0.34), (12, 0.33), (13, 0.33)],
        ]
        anchors = np.zeros((n, n_priv), dtype=float)
        for kp, entries in enumerate(anchor_specs):
            for dim_idx, weight in entries:
                anchors[:, kp] += weight * y_obs_global[:, dim_idx]

        momentum = records[-1]["sp500"] if t > 0 else 0.0

        buy_value  = np.zeros(n)
        sell_value = np.zeros(n)
        agg_total_est = np.zeros(n)

        # --- 投資家ループ ---
        for i, p in enumerate(params):
            expertise   = int(p["expertise_sector"])
            expert_mask = (sectors == expertise)
            style = float(p.get("update_style", 1.0))
            graph_quality = float(p.get("graph_quality", 1.0))
            W_i = subjective_graphs[i]

            prev_belief = belief_state[i]
            pred_state = np.empty_like(prev_belief)

            for k in range(d):
                rho_i = p["belief_rho_h"] if k >= n_pub + n_sec else p["belief_rho_s"]
                if config.use_graph:
                    graph_term = W_i @ prev_belief[:, k]
                else:
                    graph_term = 0.0
                pred_state[:, k] = p["belief_phi"] * prev_belief[:, k] + rho_i * graph_term

            updated = pred_state.copy()

            alpha_pub = config.alpha_pub_base * style * np.where(
                expert_mask, 1.0, config.alpha_pub_nonexpert_scale
            )
            alpha_pub = np.clip(alpha_pub, 0.0, config.alpha_max)
            for k in range(n_pub):
                updated[:, k] = (1.0 - alpha_pub) * pred_state[:, k] + alpha_pub * y_pub[:, k]

            alpha_sec = style * np.where(
                expert_mask, config.alpha_sec_expert, config.alpha_sec_nonexpert
            )
            alpha_sec = np.clip(alpha_sec, 0.0, config.alpha_max)
            for ks in range(n_sec):
                kg = n_pub + ks
                updated[:, kg] = (1.0 - alpha_sec) * pred_state[:, kg] + alpha_sec * y_sec[:, ks]

            if config.use_graph:
                alpha_priv = (
                    config.alpha_priv_base
                    * style
                    * graph_quality
                    * np.where(expert_mask, 1.0, config.alpha_priv_nonexpert_scale)
                )
                alpha_priv = np.clip(alpha_priv, 0.0, config.alpha_max)
                for kp in range(n_priv):
                    kg = n_pub + n_sec + kp
                    pseudo_obs = config.private_anchor_scale * p["belief_rho_h"] * rho_h_dim[kp] * (W_i @ anchors[:, kp])
                    updated[:, kg] = (1.0 - alpha_priv) * pred_state[:, kg] + alpha_priv * pseudo_obs

            belief_state[i] = updated

            total_est = updated @ dim_w
            agg_total_est += total_est

            pred_next_state = np.empty_like(updated)
            for k in range(d):
                rho_i = p["belief_rho_h"] if k >= n_pub + n_sec else p["belief_rho_s"]
                if config.use_graph:
                    graph_term = W_i @ updated[:, k]
                else:
                    graph_term = 0.0
                pred_next_state[:, k] = p["belief_phi"] * updated[:, k] + rho_i * graph_term
            pred_next = pred_next_state @ dim_w

            delta       = pred_next - total_est
            uncertainty = np.sqrt(p["obs_var"] + p["proc_var"])
            score = (
                p["value_weight"] * pred_next
                + p["trend_weight"] * delta
                - p["uncertainty_aversion"] * uncertainty
                - p["rate_sensitivity"] * rf_level
                + 0.25 * momentum
            )

            vol_ratio  = prev_market_vol_t / config.market_vol
            vol_factor = float(np.clip(1.0 + p["vol_sensitivity"] * (vol_ratio - 1.0), 0.05, 4.0))
            participation_factor = float(np.clip(vol_factor ** config.participation_vol_power, 0.10, 5.0))

            z_buy  = np.exp(np.clip( p["temperature"] * score, -20, 20)) * participation_factor
            z_sell = np.exp(np.clip(-p["loss_asymmetry"] * p["temperature"] * score, -20, 20)) * participation_factor

            # 3a. stop-loss (リスク回避型のみ)
            if config.stoploss_scale > 0.0 and prev_sp_ret < 0.0 and p["vol_sensitivity"] < 0.0:
                loss_fear = 1.0 + config.stoploss_scale * abs(prev_sp_ret) / (config.market_vol + 1e-10)
                z_sell = z_sell * float(np.clip(loss_fear, 1.0, 5.0))

            # 3b. market-wide fear: 下落後に全投資家の売りオッズを増幅
            if config.stoploss_universal_scale > 0.0 and prev_sp_ret < -config.stoploss_universal_threshold:
                fear_mult = 1.0 + config.stoploss_universal_scale * abs(prev_sp_ret) / (config.market_vol + 1e-10)
                z_sell = z_sell * float(np.clip(fear_mult, 1.0, 6.0))

            denom  = z_buy + z_sell + 1.0
            p_buy  = z_buy  / denom
            p_sell = z_sell / denom

            actions   = rng.random(n)
            buy_mask  = actions < p_buy
            sell_mask = (actions >= p_buy) & (actions < p_buy + p_sell)

            conviction = np.minimum(1.0, np.abs(score) / 0.12)
            size_frac  = p["risk_tolerance"] * vol_factor * (0.25 + conviction)
            size_frac *= rng.lognormal(mean=0.0, sigma=0.45, size=n)
            size_frac  = np.clip(size_frac, 0.0002, 0.080)

            buy_orders  = buy_mask  * cash[i]          * size_frac
            sell_orders = sell_mask * holdings[i] * firm_prices * size_frac

            total_buy = buy_orders.sum()
            if total_buy > cash[i]:
                buy_orders *= cash[i] / (total_buy + 1e-12)

            buy_value  += buy_orders
            sell_value += sell_orders
            cash[i]    += sell_orders.sum() - buy_orders.sum()
            holdings[i] += buy_orders  / firm_prices
            holdings[i] -= sell_orders / firm_prices
            holdings[i]  = np.maximum(holdings[i], 0.0)

        imbalance   = (buy_value - sell_value) / (buy_value + sell_value + 1e-9)
        total_trade = float(buy_value.sum() + sell_value.sum())

        # --- 価格形成 ---
        agg_est_mean  = agg_total_est / config.n_investors
        market_stress = float(np.mean(np.abs(agg_est_mean)) + 0.6 * abs(np.mean(imbalance)))
        if vol_trade_ewma_initialized:
            volume_ratio = vol_trade_ewma_fast / max(vol_trade_ewma_slow, 1e-12)
        else:
            volume_ratio = 1.0

        # 機構1: GJR-GARCH — 下落後に c_t sigma を増幅 (lagged down_var_ewma 使用)
        if config.gjr_scale > 0.0:
            if config.gjr_centered:
                # centered: realized_var (lagged) の半分との比率で正規化。平常時は増幅しない。
                baseline_dv = realized_var_ewma / 2.0 + 1e-12
                gjr_excess = max(down_var_ewma / baseline_dv - 1.0, 0.0)
                gjr_factor = 1.0 + config.gjr_scale * gjr_excess
            else:
                gjr_factor = 1.0 + config.gjr_scale * down_var_ewma / (config.exog_common_sigma ** 2 + 1e-12)
            current_common_sigma = config.exog_common_sigma * float(np.sqrt(max(gjr_factor, 1.0)))
        else:
            current_common_sigma = config.exog_common_sigma

        common_noise = rng.normal(0.0, current_common_sigma)
        if rng.random() < config.exog_common_jump_prob:
            common_noise += rng.standard_t(df=3) * config.exog_common_jump_sigma
        common_noise = float(np.clip(
            config.common_shock_beta * common_noise,
            -config.exog_common_clip,
             config.exog_common_clip,
        ))
        market_vol_t = prev_market_vol_t

        firm_vol = (
            config.vol_persistence * firm_vol
            + (1.0 - config.vol_persistence) * config.idio_vol
        )
        noise = rng.standard_t(df=5, size=n) * firm_vol

        # 機構2: 非対称価格インパクト — 下落後に lambda 増幅 (lagged down_var_ewma 使用)
        if config.asym_pi_scale > 0.0:
            if config.asym_pi_centered:
                # centered: realized_var/2 との比率の超過分で増幅。平常時は影響なし。
                baseline_dv = realized_var_ewma / 2.0 + 1e-12
                pi_excess = max(down_var_ewma / baseline_dv - 1.0, 0.0)
                asym_factor = 1.0 + config.asym_pi_scale * pi_excess
            else:
                asym_factor = 1.0 + config.asym_pi_scale * down_var_ewma / (config.market_vol ** 2 + 1e-12)
        else:
            asym_factor = 1.0

        # 通常の impact 計算 (activity + crash)
        impact_factor_raw = asym_factor * (1.0 + config.impact_activity_scale * max(0.0, volume_ratio - 1.0))

        # 機構4: 非対称クラッシュ — 売り超過時のみ発動
        if config.asym_crash_sell_only:
            total_sell = sell_value.sum()
            total_buy_v = buy_value.sum()
            sell_pressure_ratio = total_sell / (total_buy_v + 1e-9)
            if sell_pressure_ratio > 1.1:
                crash_excess = max(0.0, volume_ratio - config.impact_crash_threshold)
                impact_factor_raw += config.impact_crash_scale * (crash_excess ** config.impact_crash_power)
        else:
            crash_excess = max(0.0, volume_ratio - config.impact_crash_threshold)
            impact_factor_raw += config.impact_crash_scale * (crash_excess ** config.impact_crash_power)

        impact_factor = float(np.clip(impact_factor_raw, 0.25, config.impact_activity_clip))

        firm_return = config.price_impact * impact_factor * imbalance + common_noise + noise
        firm_return = np.clip(firm_return, -0.18, 0.18)

        firm_prices = firm_prices * (1.0 + firm_return)
        firm_prices = np.maximum(firm_prices, 1e-3)
        market_caps = firm_prices * shares
        weights     = market_caps / market_caps.sum()
        sp_ret      = float(np.sum(weights * firm_return))
        sp_abs      = float(sp_abs * (1.0 + sp_ret))

        records.append({
            "path_id": 0,
            "Date": dates[t].strftime("%Y-%m-%d"),
            "sp500_abs": sp_abs,
            "DGS10_abs": dgs10_abs if t > 0 else initial_dgs10,
            "sp500": sp_ret,
            "DGS10": dgs10_change,
        })

        if t in {0, 1, 2, 20, 60, 252, 504, 756, 1008, t_max - 1}:
            for j in range(n):
                rec = {"day": t, "firm_id": j, "sector": int(sectors[j])}
                for k in range(d):
                    rec[f"x{k}"] = float(x[j, k])
                rec.update({
                    "price": float(firm_prices[j]),
                    "return": float(firm_return[j]),
                    "market_weight": float(weights[j]),
                    "imbalance": float(imbalance[j]),
                    "market_vol_t": market_vol_t,
                    "down_var_ewma": down_var_ewma,
                })
                firm_records.append(rec)

        prev_common_noise = float(common_noise)
        lam_rv = config.realized_vol_lambda
        realized_var_ewma = lam_rv * realized_var_ewma + (1.0 - lam_rv) * (sp_ret ** 2)
        prev_market_vol_t = float(np.sqrt(max(realized_var_ewma, 1e-10)))
        x_prev = x.copy()

        # 取引量 EWMA 更新
        if not vol_trade_ewma_initialized:
            vol_trade_ewma_fast = vol_trade_ewma_slow = total_trade
            vol_trade_ewma_initialized = True
        else:
            lam_f = config.vol_activity_ewma_lambda
            lam_s = max(0.99, config.vol_activity_ewma_lambda)
            vol_trade_ewma_fast = lam_f * vol_trade_ewma_fast + (1.0 - lam_f) * total_trade
            vol_trade_ewma_slow = lam_s * vol_trade_ewma_slow + (1.0 - lam_s) * total_trade

        # 下落 EWMA 更新 — ループ末尾で更新し、次期の leverage 機構に使う
        neg_ret = max(-sp_ret, 0.0)
        down_var_ewma = (
            config.down_ewma_decay * down_var_ewma
            + (1.0 - config.down_ewma_decay) * neg_ret ** 2
        )
        prev_sp_ret = sp_ret

    firms_df = pd.DataFrame({
        "firm_id": np.arange(n),
        "sector": sectors,
        "initial_market_cap_weight": market_caps / market_caps.sum(),
        "true_degree": ((true_w > 0).sum(axis=1) if config.graph_topology != "zero" else np.zeros(n, dtype=int)),
    })

    config_dict = {k: getattr(config, k) for k in Config.__dataclass_fields__}

    return (
        pd.DataFrame(records),
        firms_df,
        investor_df,
        {"config": config_dict, "firm_snapshots": pd.DataFrame(firm_records)},
    )
