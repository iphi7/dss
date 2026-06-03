# Classical Models: GBM / SABR

パラメトリックな確率過程モデルによる金融時系列生成（ベースライン）。

---

## GBM（幾何ブラウン運動）

### モデル

**SP500**（対数正規分布を仮定）:
$$\log\frac{S_t}{S_{t-1}} = \left(\mu_{sp} - \frac{\sigma_{sp}^2}{2}\right)\Delta t + \sigma_{sp}\,\varepsilon_1$$

**DGS10**（算術ブラウン運動、金利なのでマイナスも許容）:
$$\Delta y_t = \mu_{dgs}\,\Delta t + \sigma_{dgs}\,\varepsilon_2$$

**クロスアセット相関**:
$$\text{corr}(\varepsilon_1, \varepsilon_2) = \rho$$

### キャリブレーション

| パラメータ | 推定方法 |
|---|---|
| $\mu_{sp}$, $\sigma_{sp}$ | SP500 対数リターンの標本平均・標準偏差 |
| $\mu_{dgs}$, $\sigma_{dgs}$ | DGS10 日次変化の標本平均・標準偏差 |
| $\rho$ | Pearson 相関係数 |

---

## SABR（Stochastic Alpha Beta Rho）

確率的ボラティリティモデル。ボラティリティ自体がランダムウォークするためボラティリティクラスタリングを捉える。

### モデル

**SP500**（$\beta=1$、対数正規 SABR）:
$$\log\frac{S_t}{S_{t-1}} = \left(\mu_{sp} - \frac{\sigma_t^2}{2}\right)\Delta t + \sigma_t\,\varepsilon_1$$
$$\sigma_t = \sigma_{t-1} \cdot \exp\!\left(\nu_{sp}\,\eta_1 - \frac{\nu_{sp}^2}{2}\right)$$
$$\text{corr}(\varepsilon_1, \eta_1) = \rho_{sp} \quad \text{（レバレッジ効果）}$$

**DGS10**（$\beta=0$、ノーマル SABR）:
$$\Delta y_t = \mu_{dgs}\,\Delta t + \sigma_{dgs,t}\,\varepsilon_2$$
$$\sigma_{dgs,t} = \sigma_{dgs,t-1} \cdot \exp\!\left(\nu_{dgs}\,\eta_2 - \frac{\nu_{dgs}^2}{2}\right)$$

**クロスアセット相関**: $\text{corr}(\varepsilon_1, \varepsilon_2) = \rho_{cross}$

### パラメータの意味

| パラメータ | 意味 | 推定方法 |
|---|---|---|
| $\sigma_0$ | 初期ボラティリティ | 直近 21 日の標準偏差 |
| $\nu$ | ボラティリティのボラティリティ（vol-of-vol） | ローリングボラ系列の対数変化の std |
| $\rho_{sp}$ | レバレッジ相関（リターン↓ → ボラ↑） | $\text{corr}(r_t, \Delta\log\sigma_{t+1})$ |
| $\rho_{cross}$ | 2資産間の革新相関 | $\text{corr}(\varepsilon_{sp}, \varepsilon_{dgs})$ |

### GBM との違い

| | GBM | SABR |
|---|---|---|
| ボラティリティ | 定数 | 確率的（vol clustering あり） |
| 尖度 | 正規（過剰尖度≈0） | fat tail（ボラの変動による） |
| レバレッジ効果 | なし | $\rho_{sp} < 0$ なら再現 |

---

## 使い方

```bash
# GBM
python classical/generate.py --model gbm --csv output.csv --n_paths 20

# SABR
python classical/generate.py --model sabr --csv output.csv --n_paths 20
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--model` | 必須 | `gbm` または `sabr` |
| `--csv` | `output.csv` | 入力データ |
| `--n_paths` | 20 | 生成パス数 |
| `--business_days` | 252 | 生成営業日数 |
| `--seed` | 42 | 乱数シード |
| `--out` | `classical/generated_paths_{model}.csv` | 出力ファイル |

### 出力形式

他モデルと共通（`path_id`, `Date`, `sp500_abs`, `DGS10_abs`, `sp500`, `DGS10`）。
