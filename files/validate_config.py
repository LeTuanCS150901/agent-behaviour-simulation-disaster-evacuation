#!/usr/bin/env python3
"""
validate_config.py — validate a stage's YAML config against its JSON Schema, plus any
cross-field mathematical constraints that JSON Schema (draft-07) cannot express directly.

Usage:
    python scripts/validate_config.py configs/stage4.yaml configs/schemas/stage4.schema.json

Must be run automatically at the start of each stage's execution (per project spec Section 5.2).
Fails fast with a specific, actionable error message — never lets an invalid parameter reach
mid-simulation.
"""
import sys
import json
import yaml
from pathlib import Path

try:
    import jsonschema
except ImportError:
    sys.exit("Missing dependency: pip install jsonschema")


# Cross-field constraints that plain JSON Schema can't express, keyed by config filename stem.
CROSS_FIELD_CHECKS = {
    "stage4": [
        (
            lambda cfg: cfg.get("model_version") == "softmax_c1_v1" and 0 < cfg["epsilon"] < 1 - cfg["q"]
            and "theta_2" not in cfg and "beta_2" not in cfg,
            "softmax_c1_v1 requires 0 < epsilon < (1 - q), truncated-normal versioning, "
            "and no free theta_2/beta_2 fields: got epsilon={epsilon}, q={q}, "
            "1-q={one_minus_q}",
        ),
        (
            lambda cfg: 0 < cfg["c"] < 1,
            "c must be strictly between 0 and 1: got c={c}",
        ),
    ],
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_schema(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate(config_path: str, schema_path: str) -> None:
    config_path, schema_path = Path(config_path), Path(schema_path)
    cfg = load_yaml(config_path)
    schema = load_schema(schema_path)

    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(cfg), key=lambda e: e.path)
    if errors:
        messages = [f"  - {'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]
        sys.exit(
            f"Config validation FAILED for {config_path} against {schema_path}:\n"
            + "\n".join(messages)
        )

    # Key cross-field checks off the *schema* filename, not the config filename — the config
    # file may be renamed/copied (e.g. for a test fixture) while the schema identity is what
    # actually determines which constraints apply.
    stage_key = schema_path.stem.replace(".schema", "")  # e.g. "stage4.schema" -> "stage4"
    for check_fn, message_template in CROSS_FIELD_CHECKS.get(stage_key, []):
        if not check_fn(cfg):
            extra = dict(cfg)
            extra["one_minus_q"] = 1 - cfg.get("q", float("nan"))
            sys.exit(
                f"Config validation FAILED for {config_path} (cross-field constraint):\n"
                f"  - {message_template.format(**extra)}"
            )

    print(f"OK: {config_path} is valid against {schema_path}"
          + (" and all cross-field constraints." if stage_key in CROSS_FIELD_CHECKS else "."))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python {sys.argv[0]} <config.yaml> <schema.json>")
    validate(sys.argv[1], sys.argv[2])
