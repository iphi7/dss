# ディレクトリ命名規則

元々の `variant_X` の記号をディレクトリ名の先頭に出す形式に統一した。

例:

```text
base7_variant_R_acf_control -> R_base7_variant_R_acf_control
goal05_variant_W_decile10_leverage -> W_goal05_variant_W_decile10_leverage
variant_A_observation_uncertainty -> A_variant_A_observation_uncertainty
```

補足: `archive_base0_initial_model` は元々 variant 記号を持たない初期スナップショットなので、先頭文字は付けていない。

| previous | current |
|---|---|
| `A_archive_base0_initial_model` | `archive_base0_initial_model` |
| `B_base1_variant_H_pub_priv` | `H_base1_variant_H_pub_priv` |
| `C_base2_variant_K_multidim` | `K_base2_variant_K_multidim` |
| `D_base3_variant_N_exog_jump_riskwealth` | `N_base3_variant_N_exog_jump_riskwealth` |
| `E_base4_variant_O_belief_ssm` | `O_base4_variant_O_belief_ssm` |
| `F_base5_variant_P_leverage` | `P_base5_variant_P_leverage` |
| `G_base6_variant_Q_longrun_stable` | `Q_base6_variant_Q_longrun_stable` |
| `H_base7_variant_R_acf_control` | `R_base7_variant_R_acf_control` |
| `I_goal01_variant_S_tail_events` | `S_goal01_variant_S_tail_events` |
| `J_goal02_variant_T_graph_jumps` | `T_goal02_variant_T_graph_jumps` |
| `K_goal03_variant_U_ultrarare_graph_jumps` | `U_goal03_variant_U_ultrarare_graph_jumps` |
| `L_goal04_variant_V_multilag_control` | `V_goal04_variant_V_multilag_control` |
| `M_goal05_variant_W_decile10_leverage` | `W_goal05_variant_W_decile10_leverage` |
| `N_experiment_L_mktcap` | `L_experiment_L_mktcap` |
| `O_experiment_M_riskpref` | `M_experiment_M_riskpref` |
| `P_variant_A_observation_uncertainty` | `A_variant_A_observation_uncertainty` |
| `Q_variant_B_neighbor_sell_pressure` | `B_variant_B_neighbor_sell_pressure` |
| `R_variant_C_exponential_memory` | `C_variant_C_exponential_memory` |
| `S_variant_C_exponential_memory_strong` | `C_variant_C_exponential_memory_strong` |
| `T_variant_C2_exponential_memory_balanced` | `C2_variant_C2_exponential_memory_balanced` |
| `U_variant_C2_exponential_memory_network` | `C2_variant_C2_exponential_memory_network` |
| `V_variant_C2_exponential_memory_network_strong` | `C2_variant_C2_exponential_memory_network_strong` |
| `W_variant_C2_exponential_memory_scaled` | `C2_variant_C2_exponential_memory_scaled` |
| `X_variant_D_no_market_common` | `D_variant_D_no_market_common` |
| `Y_variant_D_firm_common_factors` | `D_variant_D_firm_common_factors` |
| `Z_variant_D_firm_common_factors_strong` | `D_variant_D_firm_common_factors_strong` |
| `AA_variant_D_firm_common_factors_memory` | `D_variant_D_firm_common_factors_memory` |
| `AB_variant_D_firm_common_factors_tuned` | `D_variant_D_firm_common_factors_tuned` |
| `AC_variant_D_firm_common_factors_tuned2` | `D_variant_D_firm_common_factors_tuned2` |
| `AD_variant_D_firm_common_factors_final` | `D_variant_D_firm_common_factors_final` |
| `AE_variant_E_perfect_graph` | `E_variant_E_perfect_graph` |
| `AF_variant_F_no_kalman` | `F_variant_F_no_kalman` |
| `AG_variant_G_pred_score` | `G_variant_G_pred_score` |
| `AH_variant_H_pub_priv` | `H_variant_H_pub_priv` |
| `AI_variant_I_partial_obs` | `I_variant_I_partial_obs` |
| `AJ_variant_J_scale` | `J_variant_J_scale` |

更新したテキストファイル数: 120

## 追加目標ディレクトリ

| directory | note |
|---|---|
| `X_goal06_variant_X_R_issue_fix` | Rをベースにdecile10多日leverage + multi-lag ACF問題を改善する目標ディレクトリ |

| `Y_goal07_variant_Y_investor_regime` | 投資家ヘテロ性による中期ストレス反応を導入する目標ディレクトリ |
| `Z_goal08_variant_Z_jump_tail_midlev` | ジャンプ2階層・中位分位leverage・DGS10内生生成 (Round1-30, 最終候補Z117) |
| `ZA_goal09_variant_ZA_minimal_final` | Z117ベースのアブレーションで簡潔な最終モデルを作る目標ディレクトリ |
