from __future__ import annotations

import argparse
import json
import sys

from .config import load_resolved_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configuration-driven Stage 3-6 fire integration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "validate", "prepare", "headless", "gui", "parity", "finalize", "run-all",
        "replot-flow", "risk-plots",
    ):
        sub = subparsers.add_parser(name)
        sub.add_argument("--common-config", required=True)
        sub.add_argument("--runtime-config", required=True)
        if name == "replot-flow":
            sub.add_argument("--visualization-config", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "risk-plots":
        from .risk_plots import generate_risk_plots_sidecar, load_risk_plots_config

        risk_config = load_risk_plots_config(args.common_config, args.runtime_config)
        result = generate_risk_plots_sidecar(risk_config)
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return
    config = load_resolved_config(args.common_config, args.runtime_config)
    if args.command == "validate":
        from .pipeline import validation_only

        _, _, result = validation_only(config)
    elif args.command == "prepare":
        from .pipeline import build_initial_stage5, prepare_run

        result = prepare_run(config)
        result["stage5_initialization"] = build_initial_stage5(config)
    elif args.command == "replot-flow":
        from .flow_visualization import generate_clear_flow_visualizations

        result = generate_clear_flow_visualizations(config, args.visualization_config)
    else:
        from .run import finalize_run, parity_run, run_all, run_sumo_mode

        if args.command in {"headless", "gui"}:
            result = run_sumo_mode(config, args.command)
        elif args.command == "parity":
            result = parity_run(config)
        elif args.command == "finalize":
            result = finalize_run(config)
        else:
            result = run_all(config)
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
