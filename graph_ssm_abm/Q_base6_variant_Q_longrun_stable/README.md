# Q_base6_variant_Q_longrun_stable

60年分のパス生成に耐えることを目的に、`P_base5_variant_P_leverage` を長期安定化したモデルです。
短期の stylized facts だけでなく、60年を複数 seed で生成したときに指数水準が崩壊しないことを重視しています。

## 目標

- `sp500` の絶対リターン ACF をある程度再現する。
- leverage effect、すなわち過去リターンと将来二乗リターンの負相関をある程度再現する。
- 尖度を高め、fat-tail を出す。
- 60年程度の長期生成で指数がゼロ方向へ崩壊しない。
- 単一 seed ではなく複数 seed の rolling window で評価する。

## ベース構造

企業 $j$ は多次元状態 $x_{j,t}$ を持ち、企業ネットワーク $W$ 上で伝播します。

$$
x_{t+1}^{(k)}
= \phi_k x_t^{(k)}
+ \rho_k W x_t^{(k)}
+ \eta_t^{(k)}.
$$

投資家 $i$ は主観グラフ $W_i$ と belief state $b_{i,j,t}$ を持ちます。
予測ステップは

$$
\tilde b_{i,t+1}
= \phi_i b_{i,t}
+ \rho_i W_i b_{i,t}
$$

で、観測 $y_t$ により固定更新率 $\alpha_i$ で更新します。

$$
b_{i,t+1}
= (1-\alpha_i)\tilde b_{i,t+1} + \alpha_i y_t.
$$

投資家は belief から次期スコアを作り、買い・売り・待機の確率を決めます。

## Q系で追加した主な機構

### 1. score centering

長期崩壊の主因として、uncertainty や金利ペナルティが全銘柄に同じ符号で入り、全投資家が長期的に売りへ傾く問題がありました。
そこで各投資家内の平均スコアを差し引き、絶対評価だけでなく相対評価を入れました。

$$
s_{i,j,t}^{\mathrm{centered}}
= s_{i,j,t}
- \gamma_s \frac{1}{N}\sum_{j=1}^N s_{i,j,t}.
$$

今回の安定候補では $\gamma_s=0.8$ を使っています。

### 2. 弱い市場アンカー

企業別アンカーだけでは60年生成で指数水準の崩壊を止めきれなかったため、市場指数水準に弱い復元力を入れました。

$$
r_t^{\mathrm{anchor}}
= \kappa_m \tanh\left(
\frac{\log F_t - \log S_t}{h_m}
\right).
$$

ただし、これは主要な stylized facts 生成機構ではなく、長期の数値安定化のための弱い水準制約として扱います。

### 3. leverage / ACF 用のミクロ機構

下落後に売りオッズと price impact が強まる機構を残しています。

$$
\lambda_t
= \lambda_0
\left(1 + a_\lambda \cdot \mathrm{DownEWMA}_t\right),
$$

また、前期下落時には market-wide fear により売りオッズが増えます。

$$
z^{\mathrm{sell}}_{i,j,t}
\leftarrow
z^{\mathrm{sell}}_{i,j,t}
\left(1 + c_f \frac{|r_{t-1}|}{\sigma_0}\right).
$$

## 主要ファイル

- `model.py`: Config と CPU 版のベース実装。
- `model_gpu.py`: GPU 版シミュレーション本体。
- `metrics.py`: stylized facts の評価。
- `run_longrun_gpu.py`: Round1。企業別アンカー中心。
- `run_round2_gpu.py`: Round2。市場アンカー追加。
- `run_round3_gpu.py`: Round3。score centering 追加。
- `run_round4_gpu.py`: Round4。Q12ベースで leverage/ACF を戻す。
- `run_round5_gpu.py`: Round5。leverage 強化の最終探索。
- `検証メモ.md`: 各ラウンドの自動追記メモ。
- `検証レポート.md`: 結果のまとめ。
- `best_candidate_summary.csv`: 安定候補の比較。
- `all_rounds_key_metrics.csv`: Round2〜5 の主要指標一覧。

## 推奨候補

現時点の第一候補は `Q17_moreacf_fear05` です。
生成済みパスは以下にあります。

- `results_gpu_round5/Q17_moreacf_fear05/seed_1/generated_paths.csv`
- `results_gpu_round5/Q17_moreacf_fear05/seed_2/generated_paths.csv`

より leverage を重視する攻めた候補として `Q20_asym35_fear05` も残しています。
