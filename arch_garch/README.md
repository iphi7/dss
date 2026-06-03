# arch_garch – ARCH / GARCH によるパス生成

古典的な計量経済モデルである ARCH と GARCH を SP500・DGS10 にキャリブレーションし、
他モデルと同じ CSV 形式でパスを生成する。

## モデル概要

### ARCH(q)（Engle, 1982）

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \, r_{t-i}^2$$

今日のボラは「過去 $q$ 日間の二乗リターンの加重平均」で決まる。

### GARCH(p,q)（Bollerslev, 1986）

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \, r_{t-i}^2 + \sum_{j=1}^{p} \beta_j \, \sigma_{t-j}^2$$

GARCH(1,1) がデファクトスタンダード。$\beta$ が大きいほどボラの持続性が高い。

### 2変量相関の扱い

SP500 と DGS10 をそれぞれ独立にキャリブレーションし、標準化残差の相関行列を
Cholesky 分解して乱数を変換することで系列間の相関構造を再現する。

$$\mathbf{z}_t^{\text{cor}} = L \, \mathbf{z}_t^{\text{ind}}, \quad L = \text{Cholesky}(\hat{\Sigma}_\varepsilon)$$

## キャリブレーション結果（全期間 1966–2026）

| パラメータ | SP500 | DGS10 |
|---|---|---|
| ω | 0.0147 (%²) | 0.1315 (%²) |
| α₁ | 0.089 | 0.057 |
| β₁ | 0.898 | 0.942 |
| α+β | 0.987 | 0.999 |
| 残差相関 | −0.052 | ← |

$\alpha + \beta \approx 1$（IGARCH に近い）→ ボラの持続性が非常に高い。

## 使い方

```bash
# GARCH(1,1)
python arch_garch/generate.py --model garch --n_paths 20 --business_days 252

# ARCH(5)
python arch_garch/generate.py --model arch --q 5 --n_paths 20 --business_days 252
```

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `generated_paths_garch.csv` | GARCH(1,1) 生成パス |
| `generated_paths_arch.csv` | ARCH(5) 生成パス |

## 評価結果サマリー（直近 2023–2025 との乖離）

GARCH(1,1) はボラクラスタリング（|r| ACF）を適度に再現するが、
条件付き正規分布の仮定から尖度が過小評価される（γ₂ ≈ 3.5 vs 実データ ≈ 18.7）。
レバレッジ効果は $r_{t-1}^2$（符号を無視）を使うため再現できない。
