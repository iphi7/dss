# X_goal06_variant_X_R_issue_fix

`R_base7_variant_R_acf_control` をベースに、以下を同時に改善するための目標ディレクトリ。

- decile10 leverage の多日形状: 1d/3d が強すぎず、10d/20d が弱すぎないこと
- `|r|` / `r^2` の multi-lag ACF が実データから大きく外れないこと
- 尖度・tail を大きく壊さないこと

運用方針は「1アイデア1ディレクトリ」ではなく、この目標ディレクトリ内で複数アイデアを比較する。
