from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_display_text(value: Any, field_name: str) -> str:
    """Recover UTF-8 text that was repeatedly decoded as Windows-1252.

    Runtime configuration bytes are provenance inputs and must not be edited
    after a prepared run.  Display-only strings from older Windows-authored
    YAML files can nevertheless contain multiple layers of mojibake.  Decode
    only while each complete Windows-1252 -> UTF-8 reversal is valid, and
    reject text that still contains characteristic corruption markers.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    text = value
    for _ in range(8):
        try:
            repaired = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == text:
            break
        text = repaired
    if any(marker in text for marker in ("Ã", "Â", "â€", "Æ’")):
        raise ValueError(f"{field_name} contains unrecoverable text-encoding corruption: {value!r}")
    return text


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Schema must be a JSON object: {path}")
    return value


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; received {value!r}")
    return number


def _resolve(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class ResolvedIntegrationConfig:
    repository_root: Path
    common_path: Path
    runtime_path: Path
    common: dict[str, Any]
    runtime: dict[str, Any]
    common_sha256: str
    runtime_sha256: str
    logical_sha256: str
    output_root: Path
    handoff_directory: Path
    network_path: Path
    stage_configs: dict[str, Path]
    rng_seeds: dict[str, int]

    @property
    def shared(self) -> dict[str, Any]:
        return self.common["shared_contract"]

    @property
    def role(self) -> dict[str, Any]:
        return effective_role(self.common["current_codebase_engineer_contract"], self.runtime.get("demand_override"))

    @property
    def contract_role(self) -> dict[str, Any]:
        return self.common["current_codebase_engineer_contract"]

    @property
    def demand_override_provenance(self) -> dict[str, Any] | None:
        override = self.runtime.get("demand_override")
        if not override or not override.get("enabled", False):
            return None
        supplied = [name for name in ("demand", "shelters") if name in override]
        return {
            "enabled": True,
            "reason": override["reason"],
            "supplied_sections": supplied,
            "departure_generation_inherited": "demand" in override and "departure_generation" not in override["demand"],
            "contract_values": {
                "demand": deepcopy(self.contract_role["demand"]),
                "shelters": deepcopy(self.contract_role["shelters"]),
            },
            "effective_values": {
                "demand": deepcopy(self.role["demand"]),
                "shelters": deepcopy(self.role["shelters"]),
            },
            "common_contract_sha256": self.common_sha256,
        }

    def output_path(self, *parts: str) -> Path:
        return self.output_root.joinpath(*parts)

    @property
    def execution_modes(self) -> dict[str, bool]:
        """Return explicit execution modes, with legacy all-mode compatibility."""
        configured = self.runtime["execution"].get("modes")
        if configured is None:
            return {"headless_enabled": True, "gui_enabled": True, "parity_enabled": True}
        return {name: bool(value) for name, value in configured.items()}


@dataclass(frozen=True)
class IntegrationRunContext:
    config: ResolvedIntegrationConfig
    phase: str
    output_root: Path
    immutable_handoff_hashes_before: dict[str, str]


def _validate_common_cross_fields(common: dict[str, Any]) -> None:
    shared = common["shared_contract"]
    role = common["current_codebase_engineer_contract"]
    clock = shared["clock"]
    start = _finite(clock["simulation_start_seconds"], "simulation_start_seconds")
    end = _finite(clock["simulation_end_seconds"], "simulation_end_seconds")
    fire_step = _finite(clock["fire_update_seconds"], "fire_update_seconds")
    route_step = _finite(clock["route_update_seconds"], "route_update_seconds")
    sumo_step = _finite(clock["sumo_step_seconds"], "sumo_step_seconds")
    if end <= start or fire_step <= 0 or route_step <= 0 or sumo_step <= 0:
        raise ValueError("Clock end and update intervals must be positive and ordered")
    if not math.isclose(fire_step, route_step):
        raise ValueError("Fire and route update clocks must be aligned")
    if not math.isclose((fire_step / sumo_step) % 1.0, 0.0, abs_tol=1e-12):
        raise ValueError("SUMO step must divide every fire update boundary")
    derived_snapshots = int(round((end - start) / fire_step)) + 1
    if derived_snapshots != int(clock["expected_fire_snapshot_count"]):
        raise ValueError("Expected snapshot count disagrees with the configured inclusive clock")
    expected_time = clock["expected_fire_snapshot_times_seconds"]
    if (expected_time["start"], expected_time["stop"], expected_time["step"]) != (start, end, fire_step):
        raise ValueError("Expected fire snapshot range disagrees with the active clock")
    grid = shared["fire"]["grid"]
    if int(grid["rows"]) * int(grid["columns"]) != int(grid["expected_cell_count"]):
        raise ValueError("Grid rows*columns must equal expected_cell_count")
    if int(grid["expected_cell_count"]) != int(shared["validation_targets"]["cells_per_snapshot"]):
        raise ValueError("Grid and validation cell counts disagree")
    sources = shared["fire"]["ignition"]["sources"]
    if len({source["source_id"] for source in sources}) != len(sources):
        raise ValueError("Ignition source identifiers must be unique")
    if len({source["edge_id"] for source in sources}) != len(sources):
        raise ValueError("Ignition edge identifiers must be unique")
    mapping = shared["hazard"]["state_to_cell_hazard"]
    if set(mapping) != set(shared["fire"]["allowed_runtime_states"]):
        raise ValueError("Cell-hazard mapping must cover exactly the allowed runtime states")
    if any(not 0.0 <= _finite(v, f"state_to_cell_hazard.{k}") <= 1.0 for k, v in mapping.items()):
        raise ValueError("Every cell-hazard value must lie in [0,1]")
    _validate_demand_shelters(role["demand"], role["shelters"], "common contract")
    if shared["visualization"]["visualization_file_is_scientific_input"]:
        raise ValueError("The SUMO additional file must never be a scientific input")


def _validate_demand_shelters(demand_config: dict[str, Any], shelter_config: dict[str, Any], source: str) -> None:
    demand = sum(int(item["num_cars"]) for item in demand_config["origins"])
    capacity = sum(int(item["capacity"]) for item in shelter_config["destinations"])
    declared_demand = int(demand_config["total_vehicles"])
    declared_capacity = int(shelter_config["total_capacity"])
    if demand <= 0:
        raise ValueError(f"{source}: total origin demand must be positive; observed {demand}")
    if demand != declared_demand:
        raise ValueError(
            f"{source}: origin num_cars sum {demand} does not equal total_vehicles {declared_demand}"
        )
    if capacity != declared_capacity:
        raise ValueError(
            f"{source}: shelter capacity sum {capacity} does not equal total_capacity {declared_capacity}"
        )
    if capacity < demand:
        raise ValueError(f"{source}: shelter capacity {capacity} is below vehicle demand {demand}")


def effective_role(contract_role: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Return run inputs without mutating the hash-validated common contract."""
    if not override or not override.get("enabled", False):
        return contract_role
    role = deepcopy(contract_role)
    if "demand" in override:
        demand = deepcopy(override["demand"])
        if "departure_generation" not in demand:
            demand["departure_generation"] = deepcopy(contract_role["demand"]["departure_generation"])
        role["demand"] = demand
    if "shelters" in override:
        role["shelters"] = deepcopy(override["shelters"])
    return role


def _validate_overridden_edges(network_path: Path, vehicle_class: str, override: dict[str, Any] | None) -> None:
    if not override or not override.get("enabled", False):
        return
    import sumolib

    net = sumolib.net.readNet(str(network_path))
    configured: list[tuple[str, str]] = []
    if "demand" in override:
        configured.extend((str(item["edge_id"]), "demand_override.demand.origins") for item in override["demand"]["origins"])
    if "shelters" in override:
        configured.extend((str(item["edge_id"]), "demand_override.shelters.destinations") for item in override["shelters"]["destinations"])
    for edge_id, source in configured:
        try:
            edge = net.getEdge(edge_id)
        except KeyError as exc:
            raise ValueError(f"Unknown overridden SUMO edge {edge_id!r} in {source}") from exc
        if edge.getFunction() == "internal":
            raise ValueError(f"Overridden SUMO edge {edge_id!r} in {source} is internal, not a normal road edge")
        if not any(lane.allows(vehicle_class) for lane in edge.getLanes()):
            raise ValueError(f"Overridden SUMO edge {edge_id!r} in {source} does not permit {vehicle_class!r}")


def _derive_seed(base_seed: int, spawn_key: list[int]) -> int:
    import numpy as np

    state = np.random.SeedSequence(base_seed, spawn_key=tuple(spawn_key)).generate_state(1, dtype="uint32")
    return int(state[0])


def load_resolved_config(common_config: str | Path, runtime_config: str | Path) -> ResolvedIntegrationConfig:
    runtime_path = Path(runtime_config).resolve()
    common_path = Path(common_config).resolve()
    root = Path.cwd().resolve()
    runtime = _load_yaml(runtime_path)
    runtime_schema_path = _resolve(root, runtime.get("runtime_schema", ""))
    jsonschema.Draft202012Validator(_load_json(runtime_schema_path)).validate(runtime)
    configured_common = _resolve(root, runtime["common_contract"]["path"])
    if configured_common != common_path:
        raise ValueError(f"CLI common config {common_path} differs from runtime reference {configured_common}")
    common = _load_yaml(common_path)
    common_schema_path = _resolve(root, runtime["common_contract"]["schema"])
    jsonschema.Draft202012Validator(_load_json(common_schema_path)).validate(common)
    common_hash = sha256_file(common_path)
    if common_hash != runtime["common_contract"]["expected_sha256"]:
        raise ValueError(f"Common-config SHA-256 mismatch: expected {runtime['common_contract']['expected_sha256']}, got {common_hash}")
    _validate_common_cross_fields(common)
    modes = runtime["execution"].get("modes")
    if runtime["runtime_contract_version"] == "1.2" and modes is None:
        raise ValueError("Runtime contract 1.2 requires execution.modes")
    if modes is not None:
        if not modes["headless_enabled"]:
            raise ValueError("The Stage 3-6 integration requires headless execution")
        if modes["parity_enabled"] and not (modes["headless_enabled"] and modes["gui_enabled"]):
            raise ValueError("Parity requires both headless and GUI execution modes")
    replacement = runtime["reporting"].get("contract_replacement")
    if replacement is not None:
        archive_path = _resolve(root, replacement["archive_path"])
        previous_path = _resolve(root, replacement["previous_engineer_copy_path"])
        if not archive_path.is_file() or sha256_file(archive_path) != replacement["displaced_sha256"]:
            raise ValueError("Archived displaced common contract is missing or has the wrong SHA-256")
        if archive_path.stat().st_size != int(replacement["displaced_size_bytes"]):
            raise ValueError("Archived displaced common-contract size differs from runtime provenance")
        if not previous_path.is_file() or _load_yaml(archive_path) != _load_yaml(previous_path):
            raise ValueError("Archived displaced contract is not semantically identical to the previous engineer copy")
    evolution = runtime["visualization"].get("route_risk_fire_evolution")
    if runtime["runtime_contract_version"] == "1.1" and evolution is None:
        raise ValueError("Runtime contract 1.1 requires visualization.route_risk_fire_evolution")
    if evolution is not None:
        sumo_step = _finite(common["shared_contract"]["clock"]["sumo_step_seconds"], "sumo_step_seconds")
        recording = _finite(evolution["recording_interval_seconds"], "recording_interval_seconds")
        if not math.isclose(recording, sumo_step, abs_tol=float(evolution["numerical_tolerance"]), rel_tol=0.0):
            raise ValueError("route_risk_fire_evolution recording interval must equal sumo_step_seconds")
        limits = [float(value) for value in evolution["route_risk_axis_limits"]]
        if limits != [0.0, 1.0]:
            raise ValueError("Route-risk visualization axis must be exactly [0,1]")
        if Path(evolution["figure_filename"]).suffix.lower() != f".{evolution['output_format']}":
            raise ValueError("Route-risk figure filename extension differs from configured output_format")
    stage_configs = {name: _resolve(root, value) for name, value in runtime["stage_configs"].items()}
    for name, path in stage_configs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Configured {name} file does not exist: {path}")
    stage3 = _load_yaml(stage_configs["stage3"])
    backend = runtime["stage3_backend"]
    if stage3.get("solver") != backend["historical_declared_solver"]:
        raise ValueError("Stage 3 historical solver disclosure does not match the declared config")
    if not backend["authorized_substitution"]:
        raise ValueError("The active Stage 3 backend substitution must be explicitly authorized")
    base_seed = int(common["shared_contract"]["reproducibility"]["random_seed"])
    raw_sumo_seed = _derive_seed(base_seed, runtime["rng"]["sumo_stream"]["spawn_key"])
    sumo_min = int(runtime["sumo"]["random_seed_min"])
    sumo_max = int(runtime["sumo"]["random_seed_max"])
    if sumo_max < sumo_min:
        raise ValueError("SUMO random_seed_max must be >= random_seed_min")
    effective_sumo_seed = sumo_min + raw_sumo_seed % (sumo_max - sumo_min + 1)
    rng_seeds = {
        runtime["rng"]["stage4_stream"]["name"]: base_seed,
        runtime["rng"]["stage5_stream"]["name"]: _derive_seed(base_seed, runtime["rng"]["stage5_stream"]["spawn_key"]),
        runtime["rng"]["sumo_stream"]["name"]: effective_sumo_seed,
    }
    network_path = _resolve(root, common["shared_contract"]["network"]["file"])
    effective = effective_role(common["current_codebase_engineer_contract"], runtime.get("demand_override"))
    _validate_demand_shelters(effective["demand"], effective["shelters"], "effective demand override")
    _validate_overridden_edges(network_path, common["shared_contract"]["network"]["vehicle_class"], runtime.get("demand_override"))
    handoff_directory = _resolve(root, runtime["handoff"]["directory"])
    output_root = _resolve(root, runtime["execution"]["output_root"])
    runtime_hash = sha256_file(runtime_path)
    logical = logical_hash({"common": common, "runtime": runtime, "rng_seeds": rng_seeds})
    return ResolvedIntegrationConfig(
        repository_root=root,
        common_path=common_path,
        runtime_path=runtime_path,
        common=common,
        runtime=runtime,
        common_sha256=common_hash,
        runtime_sha256=runtime_hash,
        logical_sha256=logical,
        output_root=output_root,
        handoff_directory=handoff_directory,
        network_path=network_path,
        stage_configs=stage_configs,
        rng_seeds=rng_seeds,
    )
