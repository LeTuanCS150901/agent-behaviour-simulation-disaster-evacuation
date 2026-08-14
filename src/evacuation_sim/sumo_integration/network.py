from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _require_geopandas():
    try:
        import geopandas as gpd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Network coverage diagnostics require geopandas and shapely."
        ) from exc
    return gpd


def run_netconvert(stage_cfg: dict, output_net: Path, log_path: Path) -> dict:
    if stage_cfg["network_source_mode"] != "osm_xml":
        raise ValueError(f"Unsupported network_source_mode: {stage_cfg['network_source_mode']}")

    osm_path = Path(stage_cfg["osm_input_path"])
    if not osm_path.exists():
        raise FileNotFoundError(
            f"OSM input file not found: {osm_path}. Stage 1B does not download data; "
            "provide a local Toulouse OSM XML file before real network validation."
        )

    binary = stage_cfg["netconvert_binary"]
    if shutil.which(binary) is None and not Path(binary).exists():
        raise FileNotFoundError(f"netconvert binary not found: {binary}")

    output_net.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        binary,
        "--osm-files",
        str(osm_path),
        "--output-file",
        str(output_net),
        "--keep-edges.by-vclass",
        "passenger",
        "--remove-edges.isolated",
        "true",
        "--geometry.remove",
        "true",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"netconvert failed with exit code {result.returncode}; see {log_path}")
    return {"command": command, "returncode": result.returncode}


def load_sumo_network(net_path: Path):
    try:
        import sumolib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sumolib is required to validate data/toulouse.net.xml.") from exc
    if not net_path.exists():
        raise FileNotFoundError(f"SUMO network file not found: {net_path}")
    return sumolib.net.readNet(str(net_path))


def summarize_sumo_network(net, expected_crs: str) -> dict:
    bbox = net.getBoundary()
    projection = getattr(net, "getGeoProj", lambda: None)()
    location_offset = getattr(net, "getLocationOffset", lambda: None)()
    projection_text = projection or ""
    crs_verified = expected_crs in projection_text if projection_text else False
    return {
        "node_count": len(net.getNodes()),
        "edge_count": len(net.getEdges()),
        "bbox": list(bbox),
        "projection": projection,
        "location_offset": location_offset,
        "expected_crs": expected_crs,
        "crs_verified_against_base_config": crs_verified,
        "crs_check_note": (
            "SUMO projection metadata contains the expected CRS."
            if crs_verified
            else "SUMO projection metadata did not explicitly verify the expected CRS."
        ),
    }


def write_network_coverage_map(base_cfg: dict, stage_cfg: dict, net_summary: dict, output_dir: Path) -> dict:
    gpd = _require_geopandas()
    import matplotlib.pyplot as plt
    from shapely.geometry import box  # type: ignore

    study_area = gpd.read_file(stage_cfg["study_area_bbox_path"])
    if study_area.crs is None:
        raise ValueError("study area has no CRS. Cannot produce network coverage map.")
    if str(study_area.crs) != base_cfg["crs"]:
        study_area = study_area.to_crs(base_cfg["crs"])

    xmin, ymin, xmax, ymax = net_summary["bbox"]
    net_bbox = gpd.GeoDataFrame(
        [{"name": "sumo_network_bbox"}],
        geometry=[box(xmin, ymin, xmax, ymax)],
        crs=base_cfg["crs"],
    )
    study_union = study_area.geometry.union_all()
    bbox_union = net_bbox.geometry.union_all()
    coverage_ratio = float(study_union.intersection(bbox_union).area / study_union.area)

    fig, ax = plt.subplots(figsize=(9, 9))
    study_area.boundary.plot(ax=ax, color="black", linewidth=1, label="Study area")
    net_bbox.boundary.plot(ax=ax, color="#1b9e77", linewidth=1.5, label="SUMO network bbox")
    ax.set_title("Stage 1 Network Coverage")
    ax.set_axis_off()
    fig.tight_layout()
    map_path = output_dir / "network_coverage_map.png"
    fig.savefig(map_path, dpi=160)
    plt.close(fig)
    return {"coverage_ratio_bbox_intersection": coverage_ratio, "network_coverage_map": str(map_path)}


def run_stage1b(base_cfg: dict, stage_cfg: dict, output_dir: str | Path = "outputs/stage1") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net_path = Path(base_cfg["sumo_network_file"])
    log_path = output_dir / "netconvert.log"

    result = run_netconvert(stage_cfg, net_path, log_path)
    net = load_sumo_network(net_path)
    net_summary = summarize_sumo_network(net, base_cfg["crs"])
    coverage = write_network_coverage_map(base_cfg, stage_cfg, net_summary, output_dir)
    summary = {
        "stage": "stage1b",
        "status": "complete",
        "netconvert": result,
        "network": net_summary,
        "coverage": coverage,
        "outputs": {
            "sumo_network": str(net_path),
            "netconvert_log": str(log_path),
            "network_coverage_map": coverage["network_coverage_map"],
        },
    }
    (output_dir / "network_summary.json").write_text(json.dumps(net_summary | coverage, indent=2), encoding="utf-8")
    (output_dir / "stage1b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
