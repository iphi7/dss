# SigCWGAN

Signature-based Conditional Wasserstein GAN for financial time series generation.

Reference: Ni et al. (2023) "Sig-Wasserstein GANs for Time Series Generation"
(arXiv:2006.05421)

## Overview

SigCWGAN は **path signature** を用いて時系列分布を記述する生成モデルです。
Wasserstein 距離の代わりに、截断シグネチャ（truncated signature）で計算される代理距離
を最小化することで、確率過程の分布全体を捉えます。

### Path Signature とは

$d$ 次元経路 $X = (X_0, \ldots, X_T)$ の level-$k$ シグネチャ係数：

$$S^{i_1,\ldots,i_k}_{0:T} = \sum_{1 \le t_1 < \cdots < t_k \le T} \Delta X^{i_1}_{t_1} \cdots \Delta X^{i_k}_{t_k}$$

截断シグネチャ（次数 $\le K$）の次元は $\sum_{k=1}^{K} d^k$。

本実装では:
- 前処理: time augmentation + cumsum embedding（returns → 経路）+ lead-lag 変換（選択的）
- 次数: $K=3$（$d=2$ なら dim = 2 + 4 + 8 = 14, time-aug 後 $d'=3$ なら dim = 3 + 9 + 27 = 39）

### アルゴリズム (Algorithm 1)

**Step 1 (one-off):** 実データから線形回帰 $\hat{L}$ を fit

$$\hat{L}: \mathbb{R}^{d_\text{sig\_past}} \to \mathbb{R}^{d_\text{sig\_future}}, \quad \hat{L} = \arg\min_L \|L(S_\text{past}) - S_\text{future}\|_F^2 + \lambda \|L\|_F^2$$

**Step 2 (Generator training):** 生成 path のシグネチャが実データの線形写像に一致するよう学習

$$\mathcal{L} = \left\| \hat{L}(S_\text{past}) - \mathbb{E}_{Z}\left[S^\text{future}_\text{fake}\right] \right\|_2$$

期待値は $n_\text{MC}$ サンプルで近似。Generator のパラメータのみ更新（$\hat{L}$ は固定）。

### Generator アーキテクチャ (AR-FNN)

自己回帰型 FNN（Conditional AR-FNN, Appendix B.2）：

$$X_{t+1} = G\bigl(X_{t-\bar{p}+1:t},\; Z_t\bigr), \quad Z_t \sim \mathcal{N}(0, I)$$

```
input : [X_{t-p̄+1:t}.flatten(), Z_t]  ← 過去 p̄ ステップ + ノイズ
hidden: 3層 FNN with Residual + LeakyReLU
output: X_{t+1} ∈ R^2
```

## データ形式

入力: `output.csv` の `sp500`（日次リターン）と `DGS10`（金利日次変化）列を使用。

出力 `generated_paths.csv`:

| 列 | 内容 |
|---|---|
| `path_id` | パス番号 (0-indexed) |
| `Date` | 営業日 |
| `sp500_abs` | S&P 500 水準 |
| `DGS10_abs` | 10 年金利水準 |
| `sp500` | S&P 500 日次リターン |
| `DGS10` | 10 年金利日次変化 |

## 使い方

```bash
# 学習（early stopping patience=15）
python sigcwgan/train.py train --csv output.csv --epochs 200

# パス生成
python sigcwgan/train.py generate --csv output.csv --n_paths 20 --business_days 252
```

### 主な学習オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--p_bar` | 20 | 過去窓長（AR の入力長） |
| `--q_bar` | 5 | 未来窓長（生成ターゲット長） |
| `--deg_past` | 3 | 過去パスの signature 次数 |
| `--deg_future` | 3 | 未来パスの signature 次数 |
| `--n_mc` | 10 | Monte-Carlo サンプル数（期待値近似） |
| `--noise_dim` | 5 | 生成ノイズの次元 |
| `--ridge` | 1e-4 | 線形回帰の ridge 係数 |
| `--patience` | 15 | Early stopping patience |

## スコア（train / val）

**train score**: $\mathcal{L}_\text{train}$（training バッチ平均）

**val score**: validation set の $\mathcal{L}$（$\hat{L}$ は train fit で固定、val データの sig\_past に適用）

スコアが小さいほど生成分布が実データ分布に近い。

## 参照論文

- Ni et al. (2023). Sig-Wasserstein GANs for Time Series Generation. arXiv:2006.05421
