"""
O_base4_variant_O_belief_ssm/run.py

Kalman filterなしで、各投資家が企業ごとの多次元 belief_state を持つSSM版。
N19のリスク選好・取引量依存インパクトを維持し、投資家推定部分を
「予測 + 固定alpha観測更新」に変更する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import summarize_stylized_facts
from model import Config, simulate_market


def n19_base_config(**overrides) -> Config:
    params = dict(
        vol_sensitivity_mean=0.80,
        vol_sensitivity_std=0.80,
        wealth_sigma=1.20,
        wealth_vol_corr=1.20,
        participation_vol_power=1.8,
        price_impact=0.080,
        impact_activity_scale=3.0,
        impact_activity_clip=4.0,
        exog_common_sigma=0.0060,
        exog_common_jump_prob=0.001,
        exog_common_jump_sigma=0.015,
        realized_vol_lambda=0.985,
    )
    params.update(overrides)
    return Config(**params)


def run_variant(hist: pd.DataFrame, config: Config, label: str, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    generated, firms, investors, aux = simulate_market(hist, config)
    generated.to_csv(outdir / "generated_paths.csv", index=False)
    firms.to_csv(outdir / "firms.csv", index=False)
    investors.to_csv(outdir / "investors.csv", index=False)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(aux["config"], f, ensure_ascii=False, indent=2)
    print(f"  done: {label}")
    return summarize_stylized_facts(generated, label)


def main() -> None:
    hist = pd.read_csv("output.csv")
    n_days = 1260
    real_tail = hist.tail(n_days).reset_index(drop=True)
    out_base = Path("graph_ssm_abm/O_base4_variant_O_belief_ssm/results")

    rows = [summarize_stylized_facts(real_tail, "real_tail")]

    rows.append(run_variant(hist, n19_base_config(), "O0_belief_ssm_base", out_base / "O0_belief_ssm_base"))

    rows.append(run_variant(hist, n19_base_config(
        graph_topology="zero", subjective_graph_mode="partial", use_graph=True,
        mktcap_degree_reference="ba_reference",
    ), "O1_zero_graph", out_base / "O1_zero_graph"))

    rows.append(run_variant(hist, n19_base_config(
        graph_topology="ba", subjective_graph_mode="perfect", use_graph=True,
    ), "O2_perfect_graph", out_base / "O2_perfect_graph"))

    rows.append(run_variant(hist, n19_base_config(
        alpha_priv_base=0.40,
        private_anchor_scale=1.50,
    ), "O3_stronger_private_update", out_base / "O3_stronger_private_update"))

    rows.append(run_variant(hist, n19_base_config(
        alpha_pub_base=0.30,
        alpha_sec_expert=0.50,
        alpha_sec_nonexpert=0.02,
        alpha_priv_base=0.35,
        private_anchor_scale=1.50,
    ), "O4_less_public_more_private", out_base / "O4_less_public_more_private"))

    rows.append(run_variant(hist, n19_base_config(
        alpha_pub_base=0.25,
        alpha_pub_nonexpert_scale=0.45,
        alpha_sec_expert=0.55,
        alpha_sec_nonexpert=0.01,
        alpha_priv_base=0.45,
        alpha_priv_nonexpert_scale=0.50,
        private_anchor_scale=2.00,
    ), "O5_graph_emphasis", out_base / "O5_graph_emphasis"))

    rows.append(run_variant(hist, n19_base_config(
        alpha_pub_base=0.25,
        alpha_pub_nonexpert_scale=0.45,
        alpha_sec_expert=0.55,
        alpha_sec_nonexpert=0.01,
        alpha_priv_base=0.45,
        alpha_priv_nonexpert_scale=0.50,
        private_anchor_scale=2.00,
        subjective_graph_mode="perfect",
    ), "O6_graph_emphasis_perfect", out_base / "O6_graph_emphasis_perfect"))

    rows.append(run_variant(hist, n19_base_config(
        alpha_pub_base=0.25,
        alpha_pub_nonexpert_scale=0.45,
        alpha_sec_expert=0.55,
        alpha_sec_nonexpert=0.01,
        alpha_priv_base=0.45,
        alpha_priv_nonexpert_scale=0.50,
        private_anchor_scale=2.00,
        graph_topology="zero", subjective_graph_mode="partial", mktcap_degree_reference="ba_reference",
    ), "O7_graph_emphasis_zero", out_base / "O7_graph_emphasis_zero"))

    summary = pd.DataFrame(rows)
    out_csv = Path("graph_ssm_abm/O_base4_variant_O_belief_ssm/comparison_summary.csv")
    summary.to_csv(out_csv, index=False)

    cols = ["label", "std_sp500", "skew_sp500", "kurt_sp500",
            "absacf1_sp500", "absacf5_sp500", "sqacf1_sp500",
            "leverage_sp500_lag1_20", "sp_dgs10_corr"]
    print("\n=== O belief SSM comparison ===")
    print(summary[cols].to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
