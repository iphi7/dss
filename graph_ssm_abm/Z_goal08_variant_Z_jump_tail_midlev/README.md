# Z_goal08_variant_Z_jump_tail_midlev

Y_goal07（投資家レジーム機構）を継承し、以下の3課題を扱う目標ディレクトリ。

1. **Y14/Y15 の中間**: volatility clustering（abs/sq ACF）を Y14 程度に保ちつつ、短期の decile10 leverage を Y15 程度まで抑える
2. **中位分位（5-9分位）leverage の回復**: Y15 では実データ（5d で −0.07〜−0.13）に対しほぼ 0 に消失していた
3. **ジャンプサイズの回復**: Y15 の max|r|=0.0800 は `firm_return_clip=0.08` が切っていた（実データ max=0.205、q999=0.070）。絶対リターン Q-Q プロットの裾を実データに近づける

---

## モデルアーキテクチャ

### 記号の定義

| 記号 | 意味 |
|---|---|
| $n$ | 企業数（80） |
| $N$ | 投資家数（60） |
| $D$ | 企業状態の次元数（20 = 公開6 + セクター8 + 非公開6） |
| $t$ | 日次時刻 |
| $j, k$ | 企業インデックス |
| $i$ | 投資家インデックス |
| $m$ | 状態次元インデックス（$1 \le m \le D$） |
| $W \in \mathbb{R}^{n \times n}$ | 真の企業間影響グラフ（BAグラフ、行正規化済み） |
| $W_i \in \mathbb{R}^{n \times n}$ | 投資家 $i$ の主観グラフ（$W$ の辺を部分保持 + 偽辺 + 行正規化） |
| $x_{j,t} \in \mathbb{R}^{D}$ | 企業 $j$ の真の潜在状態 |
| $b_{i,j,t} \in \mathbb{R}^{D}$ | 投資家 $i$ が持つ企業 $j$ の信念状態 |
| $r_{j,t}$ | 企業 $j$ の日次リターン |
| $r_t$ | 指数（SP500 相当）の日次リターン。時価総額加重 $r_t = \sum_j w_{j,t} r_{j,t}$ |
| $w_{j,t}$ | 企業 $j$ の時価総額ウェイト |

### 1. 企業状態の遷移

各状態次元 $m$ は AR(1) + ネットワーク伝播に従う：

$$
x_{j,t,m} = \phi_m x_{j,t-1,m} + \rho_m \sum_{k} W_{jk}\, x_{k,t-1,m} + \eta_{j,t,m}
$$

ここで $\phi_m \in [0.52, 0.88]$ は次元別の自己回帰係数、$\rho_m \in [0.06, 0.35]$ はネットワーク伝播係数、$\eta_{j,t,m} \sim \mathcal{N}(0, \sigma_{\eta,m}^2)$ は次元別ノイズ。第1次元のみ確率 `rare_shock_prob` で自由度3のt分布ショック（スケール `rare_shock_sigma`）が加わる。

### 2. 観測と信念更新

公開次元は momentum 付きで観測される：

$$
y^{\mathrm{pub}}_{j,t,m} = x_{j,t,m} + \kappa_m (x_{j,t,m} - x_{j,t-1,m}) + \varepsilon^{\mathrm{pub}}_{j,t,m}
$$

ここで $\kappa_m$ は momentum 係数（第1次元 0.35、他 0.15）、$\varepsilon^{\mathrm{pub}}$ は観測ノイズ（標準偏差 `obs_sigma_pub`）。セクター次元も同様（momentum なし、`obs_sigma_sec`）。非公開次元は直接観測されず、観測済み次元の線形結合（アンカー）を主観グラフ経由で擬似観測する。

投資家 $i$ の信念は予測と観測更新の2段階：

$$
b^{\mathrm{pred}}_{i,j,t,m} = \phi^{b}_i\, b_{i,j,t-1,m} + \rho^{b}_{i,m} \sum_k (W_i)_{jk}\, b_{i,k,t-1,m}
$$

$$
b_{i,j,t,m} = (1 - \alpha_{i,j,m})\, b^{\mathrm{pred}}_{i,j,t,m} + \alpha_{i,j,m}\, y_{j,t,m}
$$

ここで $\phi^b_i$ は投資家の信念自己回帰係数、$\rho^b_{i,m}$ は公開/セクター次元用 $\rho^b_{s,i}$ と非公開次元用 $\rho^b_{h,i}$ を次元に応じて使い分けたもの、$\alpha_{i,j,m} \in [0, 0.85]$ は専門セクターか否かで異なる更新率。float32 安定性のため $\phi^b_i + \rho^b_{\cdot,i} \le 0.97$ にクランプする。

### 3. 取引スコアと注文

投資家 $i$ の企業 $j$ への取引スコア：

$$
S_{i,j,t} = v_i\, \hat{p}_{i,j,t} + \tau_i (\hat{p}_{i,j,t} - \hat{e}_{i,j,t}) - u_i \sqrt{R_i + Q_i} - g_i\, \ell_t
$$

ここで $\hat{e}_{i,j,t}$ は信念の次元加重和（現在推定値）、$\hat{p}_{i,j,t}$ は1期先予測の次元加重和、$v_i$ は value 重み、$\tau_i$ は trend 重み、$u_i$ は不確実性回避度、$R_i, Q_i$ は観測・過程分散、$g_i$ は金利感応度、$\ell_t$ は実データの DGS10 水準（比率）。スコアは投資家内で平均を部分的に差し引く（`score_centering`）。

買い/売りオッズはスコアの指数型：

$$
z^{\mathrm{buy}}_{i,j,t} = \exp(\theta_i S_{i,j,t})\, \pi_{i,t}, \qquad
z^{\mathrm{sell}}_{i,j,t} = \exp(-\lambda^{\mathrm{loss}}_i \theta_i S_{i,j,t})\, \pi_{i,t}
$$

ここで $\theta_i$ は温度、$\lambda^{\mathrm{loss}}_i \in [1.0, 1.9]$ は損失非対称度、$\pi_{i,t}$ は参加率係数。注文量は資産・リスク許容度・確信度に比例し、キャッシュ制約でスケールされる。毎日、ポートフォリオ価値のごく一部（`portfolio_rebalance_rate`）を市場ウェイトへ戻す turnover を入れ、60年生成での保有固定化を防ぐ。

### 4. 参加率とY系ストレス機構

基本参加率はボラ感応度 $v^{\sigma}_i$（正=リスク選好、負=リスク回避）から決まる：

$$
\pi_{i,t} = \mathrm{clip}\left( \left[ \mathrm{clip}\left(1 + v^{\sigma}_i \left( \frac{\hat{\sigma}_{t-1}}{\sigma_0} - 1 \right),\ 0.05,\ f_{\max}\right) \right]^{\gamma},\ \pi_{\min},\ \pi_{\max} \right)
$$

ここで $\hat{\sigma}_{t-1}$ は実現ボラの EWMA、$\sigma_0$ は基準ボラ（`market_vol`）、$\gamma$ は `participation_vol_power`、$f_{\max}$ = `vol_factor_max`、$\pi_{\min}$ = `participation_floor`、$\pi_{\max}$ = `participation_cap`（Z系で上限・下限を Config 化。高ボラ→高参加→高ボラの自己維持ゲインを制御する）。

**投資家母集団の層化抽出（Z系新規、`investor_stratified`）**: iid 抽出では投資家数 60 程度のとき「大口リスク選好投資家（クジラ）」の有無が seed 次第となり、市場が取引主導ボラへ発火するか否か（レジーム構造そのもの）が母集団の抽出運で決まってしまう（Round3 で確認）。層化抽出では資産の対数とボラ感応度を分位数グリッド

$$
u_i = \frac{i + 0.5}{N}, \qquad i = 0, 1, \ldots, N-1
$$

から標準正規分布の分位関数で生成し（ペアリングの置換のみ乱数）、母集団の抽出誤差を除去する。これにより全 seed で市場が発火するようになった。

Y系: 下落からのストレス記憶 $q_t$ を持ち、投資家行動だけに入れる：

$$
q_t = \rho_q q_{t-1} + (1 - \rho_q)\, c_q \max(-r_{t-1}, 0)^2
$$

ここで $\rho_q$ = `investor_stress_decay`、$c_q$ = `investor_stress_scale`。超過ストレス $\tilde{q}_t = \mathrm{clip}(q_t / \sigma_0^2,\ 0,\ q_{\max})$ に対し、リスク選好投資家（$v^\sigma_i > 0$）は参加率・注文量を増やし、リスク回避投資家（$v^\sigma_i < 0$）は退避する：

$$
\pi_{i,t} \leftarrow \pi_{i,t}\, \frac{1 + a\, \tilde{q}_t \max(v^{\sigma}_i, 0)}{1 + d\, \tilde{q}_t \max(-v^{\sigma}_i, 0)}
$$

ここで $a$ = `risk_pref_participation_scale`、$d$ = `risk_averse_withdraw_scale`。

**Z系拡張（投資家別ストレス記憶長）**: `investor_stress_decay_max > investor_stress_decay_min > 0` のとき、ストレス記憶を投資家別にする：

$$
q_{i,t} = \rho_{q,i}\, q_{i,t-1} + (1 - \rho_{q,i})\, c_q \max(-r_{t-1}, 0)^2
$$

減衰率 $\rho_{q,i}$ は $\max(v^{\sigma}_i, 0) \cdot \omega_i$（$\omega_i$ は資産係数）の順位に基づき `decay_min`〜`decay_max` に線形配置する。リスク選好的な大口投資家ほど記憶が長く、20d horizon の leverage の持続源になる。

### 5. 価格形成

企業リターンは注文不均衡・共通ノイズ・個別ノイズ・アンカーの和：

$$
r_{j,t} = \lambda\, F_t\, I_{j,t} + c_t + \epsilon_{j,t} + A^{\mathrm{firm}}_{j,t} + A^{\mathrm{mkt}}_t
$$

各項は：

- $I_{j,t} = \dfrac{B_{j,t} - S_{j,t}}{B_{j,t} + S_{j,t} + \varepsilon}$：注文不均衡（$B_{j,t}$ = 買い金額、$S_{j,t}$ = 売り金額）
- $\lambda$ = `price_impact`：基本価格インパクト係数
- $F_t$：インパクト増幅係数（下記）
- $c_t$：市場共通ノイズ（下記）
- $\epsilon_{j,t}$：自由度5のt分布個別ノイズ（スケール `idio_vol`）
- $A^{\mathrm{firm}}_{j,t}$：企業ファンダメンタル価値への弱い平均回帰（tanh 型、クリップ付き）
- $A^{\mathrm{mkt}}_t$：指数水準の市場アンカー（同上）

$r_{j,t}$ は $\pm$`firm_return_clip` にクランプされる。**Z系ではこのクリップを 0.08 → 0.20 に拡大**（実データの max|r|=0.205 に対応）。

#### インパクト増幅 $F_t$

$$
F_t = \mathrm{clip}\Big( \Lambda_t \big[ 1 + s_a \max(0, V_t - 1) \big] + s_c \max(0, V_t - \theta_c)^{p_c},\ 0.25,\ F_{\max} \Big)
$$

ここで $V_t$ は取引金額の速い EWMA / 遅い EWMA 比（活動比率）、$s_a$ = `impact_activity_scale`、$s_c, \theta_c, p_c$ はクラッシュ項の係数・閾値・べき、$F_{\max}$ = `impact_activity_clip`。$\Lambda_t$ は非対称価格インパクト係数（下記）。

#### 非対称価格インパクト $\Lambda_t$（Z系拡張）

下落分散 EWMA を

$$
d_t = \delta\, d_{t-1} + (1 - \delta) \max(-r_{t-1}, 0)^2
$$

（$\delta$ = `down_ewma_decay`）とし、実現分散 EWMA $\hat{\sigma}^2_{t-1}$ に対する超過を

$$
e_t = \max\left( \frac{d_t}{\hat{\sigma}^2_{t-1} / 2} - \theta_a,\ 0 \right)
$$

とする。$\theta_a$ = `asym_pi_threshold`（従来は 1.0 固定。**1 未満にすると中程度の下落でも活性化し、中位分位の leverage を作る**）。さらに `asym_pi_sat` $= s_{\mathrm{sat}} > 0$ のとき飽和させる：

$$
e_t \leftarrow s_{\mathrm{sat}} \tanh(e_t / s_{\mathrm{sat}})
$$

（**極端な下落での過剰応答を抑え、最上位分位の leverage 過大を防ぐ**）。最終的に

$$
\Lambda_t = 1 + s_{\mathrm{asym}}\, e_t
$$

（$s_{\mathrm{asym}}$ = `asym_pi_scale`）。

#### 市場共通ノイズ $c_t$ とジャンプ（Z系拡張）

$$
c_t = \mathrm{clip}\left( \beta_c \left[ \zeta_t + J_t \right],\ -C,\ C \right)
$$

ここで $\beta_c$ = `common_shock_beta`、$C$ = `exog_common_clip`、$\zeta_t \sim \mathcal{N}(0, \sigma_{c,t}^2)$ は日常ノイズ、$J_t$ はジャンプ項：

$$
J_t = \begin{cases}
\sigma_J\, T_{\nu} & \text{確率 } p_J \\
0 & \text{それ以外}
\end{cases}
$$

$\sigma_J$ = `exog_common_jump_sigma`、$p_J$ = `exog_common_jump_prob`、$T_{\nu}$ は自由度 $\nu$ = `jump_df` のt分布乱数。**Z系では $\nu$ を可変にした（従来3固定）。$\nu$ を下げると同じ頻度・スケールでも裾が重くなる。**

**ジャンプのボラ連動（Z系新規、`jump_vol_coupling`）**: 静穏レジームに大ジャンプが乗ると尖度が不自然に爆発する。実市場のクラッシュは活発な市場で起きることを踏まえ、ジャンプ振幅を実現ボラに連動させる：

$$
J_t \leftarrow J_t \cdot \mathrm{clip}\left( \left( \frac{\hat{\sigma}_{t-1}}{\sigma_0} \right)^{c_v},\ 0.3,\ 3.0 \right)
$$

ここで $c_v$ = `jump_vol_coupling`（0で無効）。静穏期のジャンプを縮め、活発期の tail を保つ。

**余震機構（Z系新規）**: ジャンプ発生時にその絶対値を余震状態 $g_t$ に加算し、日次で指数減衰させる：

$$
g_t = \rho_g\, g_{t-1} + |J_t|
$$

（$\rho_g$ = `jump_aftershock_decay`）。翌日以降の日常ノイズ標準偏差を増幅する：

$$
\sigma_{c,t} = \sigma_c \left( 1 + s_g\, g_{t-1} \right)
$$

（$\sigma_c$ = `exog_common_sigma`、$s_g$ = `jump_aftershock_scale`）。これにより「大ジャンプ→数日間の高ボラ」というクラスタが生まれ、iid ジャンプが volatility clustering を壊す問題（Y17 の弱点）を緩和する。

### 6. 指数と長期安定化

指数リターンは時価総額加重 $r_t = \sum_j w_{j,t} r_{j,t}$（+ 外生ドリフト `exog_drift`、既定0）。長期の価格崩壊を防ぐため、企業ファンダメンタル価値（緩やかに成長する潜在価格）への平均回帰 $A^{\mathrm{firm}}$ と、指数水準のアンカー $A^{\mathrm{mkt}}$（`market_anchor_*`）を持つ。内部動態（$d_t$、$q_t$、$\hat{\sigma}^2_t$）は生リターンで更新する。

---

## Z系で追加した Config 項目

| 項目 | 既定値 | 意味 |
|---|---|---|
| `jump_df` | 3.0 | 共通ジャンプのt分布自由度。小さいほど裾が重い |
| `jump_aftershock_scale` | 0.0 | 余震のボラ増幅係数。0で無効 |
| `jump_aftershock_decay` | 0.60 | 余震状態の日次減衰率 |
| `jump_vol_coupling` | 0.0 | ジャンプ振幅のボラ連動指数。静穏期の尖度爆発を抑制 |
| `exog_common_jump2_prob` / `_sigma` / `jump2_df` | 0 / 0 / 5.0 | 第2層ジャンプ（中規模・年数回）。2層構造の中間裾を作る |
| `disaster_prob` / `_sigma` / `_df` / `_decay` / `_end` | 0 / 0.05 / 4 / 0.65 / 0.10 | 災害エピソード（複数日クラッシュ）。単日特大ジャンプの構造的置換で、連続する大きな日次ショックが二乗リターンの自己相関の源になる |
| `disaster_plateau` | 0 | 危機強度を 1.0 に維持する日数。実データの sq_acf が lag5 まで平坦（危機ボラの1〜2週間高止まり）に対応 |
| `disaster_mu` | 0.0 | 危機の日次ショックの平均シフト。負にすると危機週が net 下落になり、dec10 leverage を内生的に作る |
| `mega_crash_prob` / `_size` | 0 / 0.20 | メガクラッシュ（期間内1回程度・単日 −20% 級・負符号固定）。ジャンプの超特大層 |
| `mega_triggers_episode` | 0.0 | メガクラッシュが災害エピソードを誘発する強度。1987型の暴落後乱高下を作り、孤立巨大 r² による sq_acf 希釈を防ぐ |
| `jump_aftershock2_scale` / `_decay` | 0 / 0.96 | 遅い第2余震（2〜3週間スケール）。sq_acf の lag10-20 の持続源 |
| `participation_noise_sigma` | 0.0 | 参加率への日次対数正規ノイズ。平常時の滑らかなボラ持続による \|r\| の lag1 自己相関過剰を希釈する |
| `rate_regime_center` / `_width` | 4.5 / 1.5 | 株-金利相関のレジーム係数 $\kappa_t = \tanh((c-\tilde{y}_t)/w)$ の中心・幅（% 単位）。高金利で負（割引率チャネル）、低金利で正（リスクオン） |
| `rate_trend_scale` / `_window` | 0 / 250 | 金利上昇トレンド補正。実効水準 $\tilde{y} = y + s\max(y-y_{-250},0)$。低水準でも急上昇中（インフレ）は負相関側へ |
| `rate_change_score_beta` | 0.0 | 金利→株: スコアに $\beta\kappa_t\Delta y_{t-1}(g_i/0.10)$ を加算（centering 後）。$g_i$ は投資家の金利感応度 |
| `rate_price_beta` | 0.0 | 金利→株の価格直接チャネル（leverage を希釈するため非推奨、比較用） |
| `generate_dgs10` | False | DGS10 の内生生成。1966年初期値から株価と相互作用しながら日次逐次生成 |
| `dgs10_init` / `_sigma0` / `_vol_gamma` / `_df` / `_vol_lambda` | 4.63 / 0.052 / 0.35 / 5 / 0.95 | 生成金利の初期値・基準日次ボラ・水準べき・t分布自由度・分散EWMA |
| `dgs10_drift_rho` / `_drift_sigma` | 0.9995 / 8e-5 | 持続ドリフト（インフレレジーム）の AR(1)。数十年スケールの大波の源 |
| `dgs10_mr_theta` / `_mr_center` / `_min` / `_max` | 2e-5 / 5.5 / 0.4 / 16 | 弱い平均回帰と反射境界 |
| `dgs10_stock_beta` / `dgs10_change_clip` | 0 / 0.9 | 株→金利チャネル: $\Delta y \mathrel{+}= \beta_{sr}\kappa_t r_t$（当日株リターン、質への逃避）。同日相関の主たる源 |

生成 DGS10 の日次変化の完全な形（記号は上表と本文の定義に従う）:

$$
\Delta y_t = d_t + \theta(c_{\mathrm{mr}} - y_{t-1}) + \sigma_0 \left(\frac{y_{t-1}}{5}\right)^{\gamma} \sqrt{h_t}\; T_{\nu,t} + \beta_{sr}\,\kappa_t\, r_t, \qquad
d_t = \rho_d d_{t-1} + \varepsilon_t
$$

ここで $h_t = \lambda h_{t-1} + (1-\lambda)T_{\nu,t-1}^2$ は分散状態（金利のボラクラスタ）、$T_{\nu,t}$ は自由度 $\nu$ の t 分布乱数、$\varepsilon_t \sim \mathcal{N}(0,\sigma_d^2)$。処理順は「投資家が前日金利を見て取引 → 当日株リターン $r_t$ 確定 → 当日の $\Delta y_t$ を生成」で、金利→株（1日遅れ）と株→金利（同日）の双方向結合になる。
| `obs_momentum_scale` | 1.0 | 公開観測の momentum 項スケール。0 で低分位 leverage の正相関アーティファクトを除去（市場アンカー弱体化と併用） |
| `down_deadband` / `down_deadband_rel` | 0.0 | 下落記憶のデッドバンド。**非推奨** — 静穏期の発火燃料を遮断し市場が死ぬことを Round11-12 で確認 |
| `asym_pi_threshold` | 1.0 | 非対称インパクトの活性化閾値。1未満で中位分位 leverage を強化 |
| `asym_pi_sat` | 0.0 | 超過分の tanh 飽和スケール。0で無効（線形） |
| `investor_stress_decay_min` | 0.0 | 投資家別ストレス記憶長の下限。max>min>0 で有効 |
| `investor_stress_decay_max` | 0.0 | 同上限。リスク選好×大口ほど長い記憶 |
| `investor_stratified` | False | 投資家母集団の層化抽出。レジーム構造の seed 依存を除去 |
| `participation_floor` | 0.10 | 参加率の下限 |
| `vol_factor_max` | 4.0 | ボラ感応係数の上限（高温レジームの自己維持ゲイン制御） |
| `participation_cap` | 5.0 | 参加率の上限（同上） |

`firm_return_clip` は既存項目だが、Z系では 0.08 → 0.20 に拡大して運用する（ジャンプサイズ回復。Y15 では max|r| がこのクリップに張り付いていた）。

---

## 評価

`run_round*.py` が seed 1/2/3 で 60 年パスを生成し、以下を評価する:

- multi-lag ACF: raw/absolute/squared リターンの lag 1/2/3/5/10/20/60
- 分位別 leverage: 当期 |r| の10分位ごとに、将来 1/3/4/5/7/10/20 日平均 |r| との相関
  - decile10（重み 2.2）、**中位分位 5-9（重み 0.6、Z系で追加）**、decile1（重み 0.8）
- **裾分位点（Z系で追加）**: |r| の q99/q995/q999/max（絶対リターン Q-Q プロットの裾に対応）
- 尖度・標準偏差

実データ目標値（output.csv 末尾 14,882 日）:

| 指標 | 値 |
|---|---|
| std | 0.0105 |
| kurtosis | 18.75 |
| abs_acf5 / sq_acf5 | 0.289 / 0.228 |
| abs_q999 / max | 0.0704 / 0.2047 |
| dec10 leverage 3d/5d/10d/20d | −0.203 / −0.209 / −0.176 / −0.160 |
| 中位分位(5-9)平均 leverage 5d/10d | −0.109 / −0.108 |

詳細な検証経過は `検証メモ.md`、結論は `検証レポート.md` を参照。
