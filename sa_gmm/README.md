# sa_gmm – GMM 切断正規分布 + 動的レジーム確率 DDPM

## モデルの概要

`cond_random`（sign_ablation）の符号ウィンドウ部分を、
**GMM 切断正規分布からサンプルした z_t** に置き換えたモデル。
arch_gm の Gaussian Mixture ノイズ設計を DDPM に統合し、
さらに **Small LSTM によるレジーム確率の動的変動** を加えることで
ボラティリティクラスタリングを再現する。

---

## アーキテクチャ詳細

### 1. 動的スライディング σ（O(1)更新）

各時刻 t における直近 k=30 日のリターン分散を O(1) で更新：

```
mean_new = mean_old + (r[t] - r[t-k]) / k
var_new  = var_old + mean_old² + (r[t]² - r[t-k]²) / k - mean_new²
σ_t = sqrt(var_new)
```

### 2. レアケースフラグ

```
is_rare[t, d] = (|r[t,d]| > 3 * σ[t,d])  ∈ {0, 1}
```

### 3. Small LSTM（hidden=8）

is_rare 系列を時系列として処理し、動的レジーム確率を出力：

```
h_p_t = SmallLSTM(is_rare_t, h_p_{t-1})   # hidden=8
p_t   = sigmoid(Linear(8 → 1))
```

初期化: sigmoid(bias) ≈ p_base（最初の3年から推定）

**ボラクラスタリングの発生機構：**
大きい|r|が続く → is_rare=1 が続く → SmallLSTM が h_p_t を更新
→ p_t が上昇 → z_t が裾コンポーネントから来やすくなる
→ EpsilonNet が大きい|r|を生成 → |r| ACF が上がる

### 4. GMM 切断正規分布サンプリング（STE）

```
component ~ Bernoulli(p_t)  ← Straight-Through Estimator で微分可能化

z_t = { N(0, σ_n²)  truncated to  |z| < τ   (通常コンポーネント, 中央部分)
       { N(0, σ_r²)  truncated to  |z| > τ   (レアコンポーネント, 裾部分)
       τ = 3.0（正規化空間）
```

STE: `component = hard + (p_t - p_t.detach())`
前向きは離散 binary、後向きは p_t への勾配が流れる。

**GMM パラメータ（最初の3年から推定、以降固定）：**

| パラメータ | 意味 |
|---|---|
| p_base | is_rare の観測頻度（初期 bias） |
| σ_n (D,) | 通常リターン（|r| ≤ 3σ_t）の std |
| σ_r (D,) | レアリターン（|r| > 3σ_t）の std |

### 5. EpsilonNet 条件

```
cond = [h_main(64) || z_t(2) || sign_window(60)] = 126d

sign_window: 直近 30 日の実際の符号（訓練時・生成時ともに実符号）
  訓練時: データの真の符号
  生成時: 生成済みパスの符号（自己回帰）
```

sign_window の役割: 正則化（cond_random の Random Ber(1/2) と同様の効果を、
実際の符号方向情報とともに与える）

### 6. DDPM reverse

```
r̂_t ← DDPM(EpsilonNet, cond)   # 4層 ResidualBlock, hidden=64, T=100
```

---

## cond_random との比較

| 要素 | cond_random | sa_gmm |
|---|---|---|
| sign_window 内容 | Ber(1/2) ランダム | 実際の符号（自己回帰） |
| レジーム情報 | なし | z_t（GMM 切断正規分布） |
| レジーム確率 | 固定 (0.5) | p_t = f(SmallLSTM(is_rare)) |
| VolClustering 機構 | 偶発的（正則化の副作用） | 明示的（Small LSTM → p_t → z_t） |
| パラメータ数増加 | — | +SmallLSTM(8): わずか |

---

## ファイル構成

```
sa_gmm/
  dataset.py   – データ読み込み・σ計算・is_rare・GMM パラメータ推定
  model.py     – GMMSignTimeGrad, sample_gmm_z, 各サブモジュール
  train.py     – 学習スクリプト
  generate.py  – パス生成スクリプト（start_date 指定対応）
  README.md    – 本ファイル
```

---

## 使い方

```bash
# 学習
python sa_gmm/train.py

# 生成（デフォルト: データ末尾から 252 日）
python sa_gmm/generate.py

# 開始日指定
python sa_gmm/generate.py --start_date 2006-05-14 --business_days 5040

# データ先頭から 60 年分
python sa_gmm/generate.py --start_date 1966-01-03 --business_days 15120 --n_paths 20
```

---

## 主なハイパーパラメータ

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--context_length` | 252 | LSTM コンテキスト長（営業日） |
| `--pred_length` | 21 | 教師強制の予測長 |
| `--rnn_hidden` | 64 | Main LSTM 隠れ次元 |
| `--small_hidden` | 8 | Small LSTM 隠れ次元 |
| `--slide_k` | 30 | σスライドウィンドウ幅 |
| `--hidden_dim` | 64 | EpsilonNet 隠れ次元 |
| `--n_layers` | 4 | EpsilonNet ResidualBlock 数 |
| `--diff_steps` | 100 | DDPM 拡散ステップ数 |
| `--tau` | 3.0 | 切断正規分布の境界（固定） |
