# arch_gm – ARCH' (ARCH with Gaussian Mixture noise)

標準 ARCH の i.i.d. N(0,1) ノイズを **Gaussian Mixture** に置き換えたモデル。
ARCH が提供する動的ボラティリティ構造と、混合正規分布が提供する静的な fat tail を組み合わせる。

## モデルの定義

$$r_t = \sigma_t \cdot \varepsilon_t$$

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \, r_{t-i}^2 \quad \text{（ARCH(q) 条件分散）}$$

$$\varepsilon_t \sim (1-p)\,\mathcal{N}(0, \sigma_1^2) + p\,\mathcal{N}(0, \sigma_2^2), \quad (1-p)\sigma_1^2 + p\sigma_2^2 = 1$$

$\varepsilon_t$ はサンプル後に $\sqrt{(1-p)\sigma_1^2 + p\sigma_2^2}$ で割って単位分散に正規化する。
そのため $\sigma_1, \sigma_2$ の **絶対値は無効で、比率 $\sigma_2/\sigma_1$ と $p$ だけが分布の形を決める**。

## Gaussian Mixture ノイズの設計

実データ SP500 から導出した経験的パラメータ（2節参照）:

| パラメータ | 値 | 導出根拠 |
|---|---|---|
| $p$ | 0.013 | $\|r\| > 3\sigma$ の観測割合（正規分布の理論値 0.27% の約5倍） |
| $\sigma_2/\sigma_1$ | 4.5 | テール事象のスケール感（$\sigma' \approx 4.6\%$ vs $\sigma \approx 1.0\%$） |

正規化後のパラメータ（$\sigma_2/\sigma_1 = 4.5$, $p = 0.013$ のとき）:

$$\sigma_1' = \frac{1}{\sqrt{(1-p) + p \cdot 4.5^2}} \approx 0.894, \quad \sigma_2' = 4.5\,\sigma_1' \approx 4.024$$

**ε の尖度**（analytically）:

$$\gamma_2(\varepsilon) = 3\bigl[(1-p)\sigma_1'^4 + p\,\sigma_2'^4\bigr] \approx 12.1$$

大きい $p$ の近似（$\sigma_2 \gg \sigma_1$）では $\gamma_2 \approx 3/p$（[参照: stylized_facts.md §1の注意]）。

## 標準 ARCH との違い

| | ARCH | ARCH'（本モデル） |
|---|---|---|
| 条件分散 $\sigma_t^2$ | 同一（ARCH(q)） | 同一 |
| ノイズ $\varepsilon_t$ | $\mathcal{N}(0,1)$ | Gaussian Mixture |
| 無条件尖度 | ARCH のみに依存（過小評価） | ARCH + 混合ノイズの両方が寄与 |
| レバレッジ効果 | 再現不可（$r_{t-1}^2$ は符号を無視） | 同様に再現不可 |

## 評価結果（直近 2023–2025 との乖離）

| 指標 | arch_gm | arch5 | garch | 実データ |
|---|---|---|---|---|
| 尖度 γ₂ (SP500) | **≈8.2** | ≈3.2 | ≈3.5 | 18.7 |
| \|r\| ACF lag-3 | **≈0.187** | ≈0.028 | ≈0.125 | 0.220 |
| レバレッジ効果 | −0.008 | ≈0 | −0.005 | −0.134 |

尖度は ARCH・GARCH より大幅に改善。|r| ACF も最良レベル。
レバレッジ効果・歪度は ARCH 族の構造的限界で再現不可。

## 使い方

```bash
# デフォルト設定（p=0.013, ratio=4.5, q=5）
python arch_gm/generate.py

# パラメータ変更
python arch_gm/generate.py --p 0.02 --ratio 5.0 --q 3
```

### CLI オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--q` | 5 | ARCH の次数 |
| `--p` | 0.013 | GM テール確率 |
| `--k` | 0.001 | σ₁ = k（正規化されるため比率のみが有効） |
| `--ratio` | 4.5 | σ₂/σ₁ の比率 |
| `--n_paths` | 20 | 生成パス数 |
| `--business_days` | 252 | 生成営業日数 |
