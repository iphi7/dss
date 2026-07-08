# Y_goal07_variant_Y_investor_regime

`R_base7_variant_R_acf_control` / `X_goal06_variant_X_R_issue_fix` をベースに、パラメータ調整ではなく投資家行動機構を追加して改善する目標ディレクトリ。

目的:

- decile10 leverage の 1d/3d 過剰を抑えつつ、10d/20d 側を戻す
- `|r|` / `r^2` の multi-lag ACF を保つ
- tail/kurtosis を極端に壊さない

追加機構:

1. slow investor stress memory
2. risk-seeking investor activity boost
3. risk-averse investor withdrawal
4. optional contrarian buy by risk-seeking investors after stress

市場ボラを直接いじるのではなく、投資家タイプごとの参加率・注文量・買い売り傾向を変える。
