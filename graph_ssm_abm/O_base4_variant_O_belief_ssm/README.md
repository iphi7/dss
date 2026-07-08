# base4 variant O: Kalmanなし belief-state SSM

## 目的

`N_base3_variant_N_exog_jump_riskwealth` では、投資家は各時刻の観測 `y_t` からその場で企業評価を作っていた。つまり、投資家ごとの企業状態推定は時刻をまたいで保存されていなかった。

この variant O では、元々の構想に近づけるため、各投資家が企業ごとの多次元内部状態を持つ。

$$
\hat{X}^{(i)}_t \in \mathbb{R}^{N_{firm} \times d}
$$

ただし Kalman filter は使わない。代わりに、

1. 主観グラフによる予測
2. 固定された更新率 \(\alpha\) による観測ブレンド

で belief state を更新する。

## 状態更新

投資家 \(i\) の企業 \(j\)、次元 \(k\) に対する内部推定を

$$
\hat{X}^{(i)}_{j,k,t}
$$

とする。

### 予測ステップ

$$
\hat{X}^{(i)-}_{:,k,t}
=
\phi_i \hat{X}^{(i)}_{:,k,t-1}
+
\rho^{(i)}_k W_i \hat{X}^{(i)}_{:,k,t-1}
$$

ここで \(W_i\) は投資家 \(i\) の主観グラフである。

### 観測更新ステップ

観測できる成分は、固定 \(\alpha\) によって予測値と観測値をブレンドする。

$$
\hat{X}^{(i)}_{j,k,t}
=
(1-\alpha^{(i)}_{j,k})\hat{X}^{(i)-}_{j,k,t}
+
\alpha^{(i)}_{j,k}y_{j,k,t}
$$

Kalman gain は使っていない。\(\alpha\) は投資家の固定された更新戦略である。

## \(\alpha\) の設定

初期実装では、\(\alpha\) は時間変化しない固定パラメータとした。

public dims:

$$
\alpha^{pub}_{i,j}
= 0.45 \cdot u_i \cdot
\begin{cases}
1.0, & s_i=s_j,\\
0.70, & s_i\ne s_j.
\end{cases}
$$

sector dims:

$$
\alpha^{sec}_{i,j}
= u_i \cdot
\begin{cases}
0.60, & s_i=s_j,\\
0.05, & s_i\ne s_j.
\end{cases}
$$

private dims:

$$
\alpha^{priv}_{i,j}
=0.25\cdot u_i\cdot g_i\cdot
\begin{cases}
1.0, & s_i=s_j,\\
0.30, & s_i\ne s_j.
\end{cases}
$$

ここで \(u_i\) は投資家の更新癖、\(g_i\) は主観グラフの品質である。

## private anchor

private dims は直接観測されないため、public/sector 観測から固定対応表で anchor を作る。

$$
y^{anchor,k}_t
=
\sum_{\ell \in A_k} a_{k\ell}y^\ell_t
$$

その後、主観グラフで近傍集約して擬似観測を作る。

$$
\tilde{y}^{priv,k,(i)}_t
=
\rho^{(i)}_h\rho^{(k)}_h W_i y^{anchor,k}_t
$$

private belief は

$$
\hat{X}^{(i)}_{:,k,t}
=
(1-\alpha^{priv})\hat{X}^{(i)-}_{:,k,t}
+
\alpha^{priv}\tilde{y}^{priv,k,(i)}_t
$$

で更新する。

## 売買と価格形成

N19 と同じく、リスク選好型大口投資家の高ボラ時の参加率・注文量増加、および取引量依存の価格インパクトを残している。

企業リターンは

$$
r_{j,t}
= \lambda_t I_{j,t} + c_t + \epsilon_{j,t}
$$

である。

## 実行

```bash
/home/u00121/.venv/bin/python -m graph_ssm_abm.O_base4_variant_O_belief_ssm.run
```

## 出力

- `comparison_summary.csv`
- `results/<variant>/generated_paths.csv`
- `results/<variant>/firms.csv`
- `results/<variant>/investors.csv`
- `results/<variant>/config.json`


## 追加機構: 流動性クラッシュ

O12以降では、尖度不足への対策として、外生ジャンプの強化と流動性クラッシュを検証した。

流動性クラッシュでは、取引量 fast/slow 比率

$$

u_t = rac{V^{fast}_{t-1}}{V^{slow}_{t-1}}
$$

が閾値を超えた場合のみ、価格インパクトを

$$
\lambda_t
= \lambda_0\left[1 + a(
u_t-1)_+ + b(
u_t-
u_c)_+^p
ight]
$$

で非線形に増幅する。

代表結果として、O19 は kurtosis=7.47, absacf1=0.163, absacf5=0.145 となり、O10より厚い裾を回復しつつボラクラも維持した。O20 は kurtosis=9.08 まで上がったが、ACF は弱めである。
