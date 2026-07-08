"""
既存 round/config/seed を、実データ60年前の初期水準から再生成する補助スクリプト。

通常の Z 系 run_round*.py は、生成開始時の SP500 水準を output.csv の最終値
（2026年時点）に置く。一方、60年分の歴史期間と重ねて可視化・検証する場合は、
SP500 も歴史tailの先頭水準から生成した方が自然である。

例:
    python graph_ssm_abm/Z_goal08_variant_Z_jump_tail_midlev/regenerate_histinit.py \
      --round 25 --config Z98_trend10 --seed 1
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]


def _json_safe(obj):
    """meta に DataFrame/ndarray 等が混じっても軽い形で保存できるようにする。"""
    if isinstance(obj, pd.DataFrame):
        return {"type": "DataFrame", "shape": list(obj.shape), "columns": list(obj.columns)}
    if isinstance(obj, pd.Series):
        return {"type": "Series", "shape": list(obj.shape), "name": obj.name}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, help="run_round{N}_gpu.py の N")
    parser.add_argument("--config", required=True, help="CONFIGS 内のキー名。例: Z98_trend10")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-days", type=int, default=15120)
    parser.add_argument("--output-csv", default=str(PROJECT_ROOT / "output.csv"))
    parser.add_argument("--out-suffix", default="histinit")
    parser.add_argument("--plot", action="store_true", help="plot_paths.py で概要PNGも作る")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    mod = importlib.import_module(f"run_round{args.round}_gpu")

    if args.config not in mod.CONFIGS:
        keys = ", ".join(sorted(mod.CONFIGS))
        raise KeyError(f"{args.config!r} not found in CONFIGS. Available: {keys}")

    output_df = pd.read_csv(args.output_csv)
    n_days = min(args.n_days, len(output_df) - 2)
    real_tail = output_df.tail(n_days).reset_index(drop=True)

    base_cfg = mod.CONFIGS[args.config]
    params = asdict(base_cfg)
    params.update(
        seed=args.seed,
        n_days=n_days,
        initial_sp500_abs=float(real_tail.loc[0, "sp500_abs"]),
        initial_dgs10_abs=float(real_tail.loc[0, "DGS10_abs"]),
    )
    config = mod.Config(**params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} config={args.config} seed={args.seed} "
        f"initial_sp500_abs={config.initial_sp500_abs} "
        f"initial_dgs10_abs={config.initial_dgs10_abs} n_days={config.n_days}",
        flush=True,
    )

    paths, firms, investors, meta = mod.simulate_market_gpu(output_df, config, device=device)

    out_dir = ROOT / f"results_gpu_round{args.round}" / args.config / f"seed_{args.seed}_{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # raw: モデルが通常生成する「2026年以後」の日付のまま保存。
    paths.to_csv(out_dir / "generated_paths_future_dates.csv", index=False)

    # aligned: 歴史期間で重ねて見るため、Date と DGS10_abs を実データtailに揃える。
    aligned = paths.copy()
    aligned["Date"] = real_tail["Date"].to_numpy()
    aligned["DGS10_abs"] = real_tail["DGS10_abs"].to_numpy()
    if "DGS10" in aligned.columns:
        aligned["DGS10"] = aligned["DGS10_abs"].diff().fillna(0.0)
    aligned.to_csv(out_dir / "generated_paths.csv", index=False)

    firms.to_csv(out_dir / "firms.csv", index=False)
    investors.to_csv(out_dir / "investors.csv", index=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, ensure_ascii=False, indent=2)

    print(f"wrote {out_dir / 'generated_paths.csv'}")
    print(aligned[["Date", "sp500_abs", "DGS10_abs", "sp500"]].head().to_string(index=False))
    print(aligned[["Date", "sp500_abs", "DGS10_abs", "sp500"]].tail().to_string(index=False))

    if args.plot:
        import subprocess

        plot_script = PROJECT_ROOT / "plot_paths.py"
        out_png = out_dir / "paths_overview.png"
        subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "--gen",
                str(out_dir / "generated_paths.csv"),
                "--since",
                "1966-01",
                "--out",
                str(out_png),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
