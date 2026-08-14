"""Pure Stage 4 ``softmax_c1_v1`` mathematics and validation.

``W_s`` and ``W_g`` are raw dimensionless scores. ``S_s`` and ``S_g`` are
softmax shares of the non-panic mass. Only ``V_p``, ``V_s`` and ``V_g`` are
mixture probabilities.

Mathematical inputs follow NumPy broadcasting. Scalars can therefore be used
with vectors or higher-dimensional arrays. Non-broadcastable shapes fail with
an error that names every supplied shape.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import truncnorm


MODEL_VERSION = "softmax_c1_v1"
SOFTMAX_TEMPERATURE = 1.0


def _array(name: str, value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values; got {value!r}")
    return result


def _broadcast(**values: Any) -> dict[str, np.ndarray]:
    arrays = {name: _array(name, value) for name, value in values.items()}
    try:
        broadcast = np.broadcast_arrays(*arrays.values())
    except ValueError as exc:
        shapes = ", ".join(f"{name}={array.shape}" for name, array in arrays.items())
        raise ValueError(
            f"Stage 4 inputs cannot be broadcast consistently; supplied shapes: {shapes}"
        ) from exc
    return dict(zip(arrays, broadcast, strict=True))


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def _validated_curve_inputs(
    *, x: Any | None = None, epsilon: Any, c: Any, q: Any
) -> dict[str, np.ndarray]:
    values = {"epsilon": epsilon, "c": c, "q": q}
    if x is not None:
        values = {"x": x, **values}
    arrays = _broadcast(**values)
    if x is not None and np.any((arrays["x"] < 0.0) | (arrays["x"] > 1.0)):
        raise ValueError("x must lie in [0,1]")
    if np.any((arrays["c"] <= 0.0) | (arrays["c"] >= 1.0)):
        raise ValueError("c must satisfy 0 < c < 1")
    if np.any((arrays["q"] < 0.0) | (arrays["q"] > 1.0)):
        raise ValueError("q must satisfy 0 <= q <= 1")
    m = 1.0 - arrays["q"]
    if np.any((arrays["epsilon"] < 0.0) | (arrays["epsilon"] > m)):
        raise ValueError("epsilon must satisfy 0 <= epsilon <= M, where M=1-q")
    arrays["M"] = m
    return arrays


def analytical_c1_coefficients(epsilon: Any, c: Any, q: Any) -> dict[str, Any]:
    """Return analytical C1 coefficients using NumPy broadcasting."""
    a = _validated_curve_inputs(epsilon=epsilon, c=c, q=q)
    e, join, m = a["epsilon"], a["c"], a["M"]
    values = {
        "mode": MODEL_VERSION,
        "epsilon": e,
        "c": join,
        "q": a["q"],
        "M": m,
        "theta_0": e,
        "theta_1": 2.0 * (m - e) / join,
        "theta_2": (e - m) / join**2,
        "beta_0": m - m * join**2 / (1.0 - join) ** 2,
        "beta_1": 2.0 * m * join / (1.0 - join) ** 2,
        "beta_2": -m / (1.0 - join) ** 2,
    }
    return {
        name: _scalar_or_array(value) if isinstance(value, np.ndarray) else value
        for name, value in values.items()
    }


def raw_selfish_score(x: Any, epsilon: Any, c: Any, q: Any) -> float | np.ndarray:
    """Evaluate the raw dimensionless selfish score ``W_s``."""
    a = _validated_curve_inputs(x=x, epsilon=epsilon, c=c, q=q)
    x_a, e, join, m = a["x"], a["epsilon"], a["c"], a["M"]
    left = e + 2.0 * (m - e) * x_a / join + (e - m) * x_a**2 / join**2
    right = m - m * ((x_a - join) / (1.0 - join)) ** 2
    return _scalar_or_array(np.where(x_a < join, left, right))


def raw_government_score(x: Any, epsilon: Any, c: Any, q: Any) -> float | np.ndarray:
    """Evaluate raw compliance score ``W_g = 1-x-W_s``; negatives are valid."""
    a = _validated_curve_inputs(x=x, epsilon=epsilon, c=c, q=q)
    selfish = np.asarray(raw_selfish_score(a["x"], a["epsilon"], a["c"], a["q"]))
    return _scalar_or_array(1.0 - a["x"] - selfish)


def stable_softmax_shares(raw_selfish: Any, raw_government: Any) -> tuple[Any, Any]:
    """Return stable two-score softmax shares at the fixed temperature ``T=1``."""
    a = _broadcast(raw_selfish=raw_selfish, raw_government=raw_government)
    scores = np.stack([a["raw_selfish"], a["raw_government"]], axis=-1)
    maximum = np.max(scores, axis=-1, keepdims=True)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shifted = scores - maximum
        exponentials = np.exp(shifted)
        shares = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    if not np.all(np.isfinite(shares)):
        raise ValueError("Stable softmax produced non-finite shares from finite scores")
    return _scalar_or_array(shares[..., 0]), _scalar_or_array(shares[..., 1])


def panic_weight(x: Any) -> float | np.ndarray:
    x_a = _array("x", x)
    if np.any((x_a < 0.0) | (x_a > 1.0)):
        raise ValueError("x must lie in [0,1]")
    return _scalar_or_array(x_a.copy())


def selfish_weight(x: Any, selfish_softmax_share: Any) -> float | np.ndarray:
    a = _broadcast(x=x, selfish_softmax_share=selfish_softmax_share)
    if np.any((a["x"] < 0.0) | (a["x"] > 1.0)):
        raise ValueError("x must lie in [0,1]")
    if np.any((a["selfish_softmax_share"] < 0.0) | (a["selfish_softmax_share"] > 1.0)):
        raise ValueError("selfish_softmax_share must lie in [0,1]")
    return _scalar_or_array((1.0 - a["x"]) * a["selfish_softmax_share"])


def government_weight(x: Any, government_softmax_share: Any) -> float | np.ndarray:
    a = _broadcast(x=x, government_softmax_share=government_softmax_share)
    if np.any((a["x"] < 0.0) | (a["x"] > 1.0)):
        raise ValueError("x must lie in [0,1]")
    if np.any((a["government_softmax_share"] < 0.0) | (a["government_softmax_share"] > 1.0)):
        raise ValueError("government_softmax_share must lie in [0,1]")
    return _scalar_or_array((1.0 - a["x"]) * a["government_softmax_share"])


def compute_mixture_weights(x: Any, epsilon: Any, c: Any, q: Any) -> dict[str, Any]:
    """Compute raw W scores, normalized S shares, and final V probabilities."""
    a = _validated_curve_inputs(x=x, epsilon=epsilon, c=c, q=q)
    raw_s = raw_selfish_score(a["x"], a["epsilon"], a["c"], a["q"])
    raw_g = raw_government_score(a["x"], a["epsilon"], a["c"], a["q"])
    share_s, share_g = stable_softmax_shares(raw_s, raw_g)
    v_p = panic_weight(a["x"])
    v_s = selfish_weight(a["x"], share_s)
    v_g = government_weight(a["x"], share_g)
    return {
        "raw_selfish_score": raw_s,
        "raw_government_score": raw_g,
        "selfish_softmax_share": share_s,
        "government_softmax_share": share_g,
        "panic_weight": v_p,
        "selfish_weight": v_s,
        "government_weight": v_g,
    }


def _validate_distribution(name: str, values: Any, tolerance: float) -> np.ndarray:
    array = _array(name, values)
    if array.ndim < 1 or array.shape[-1] == 0:
        raise ValueError(f"{name} must have a non-empty shelter axis")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    sums = np.sum(array, axis=-1)
    if not np.all(np.abs(sums - 1.0) <= tolerance):
        raise ValueError(
            f"{name} must sum to one within probability_tolerance={tolerance}; "
            f"maximum error={float(np.max(np.abs(sums - 1.0)))}"
        )
    return array


def compose_shelter_probabilities(
    government_distribution: Any,
    selfish_distribution: Any,
    panic_distribution: Any,
    *,
    government_mixture_weight: Any,
    selfish_mixture_weight: Any,
    panic_mixture_weight: Any,
    probability_tolerance: float,
) -> np.ndarray:
    """Compose final shelter probabilities using only final V weights.

    Component distributions broadcast across their leading dimensions; the last
    dimension is always the shelter axis. Mixture weights must be scalar or
    broadcastable to the component leading dimensions.
    """
    tolerance = float(_array("probability_tolerance", probability_tolerance))
    if tolerance <= 0.0:
        raise ValueError("probability_tolerance must be finite and > 0")
    pg = _validate_distribution("P_g", government_distribution, tolerance)
    ps = _validate_distribution("P_s", selfish_distribution, tolerance)
    pp = _validate_distribution("P_p", panic_distribution, tolerance)
    try:
        pg, ps, pp = np.broadcast_arrays(pg, ps, pp)
    except ValueError as exc:
        raise ValueError(
            "P_g, P_s and P_p cannot be broadcast consistently; supplied shapes: "
            f"P_g={pg.shape}, P_s={ps.shape}, P_p={pp.shape}"
        ) from exc
    shelter_count = pg.shape[-1]
    uniform = np.full(shelter_count, 1.0 / shelter_count)
    if not np.all(np.abs(pp - uniform) <= tolerance):
        raise ValueError("P_p must be uniform over the explicitly eligible shelters")

    leading_shape = pg.shape[:-1]
    weights = _broadcast(
        government_mixture_weight=government_mixture_weight,
        selfish_mixture_weight=selfish_mixture_weight,
        panic_mixture_weight=panic_mixture_weight,
    )
    weight_arrays: dict[str, np.ndarray] = {}
    for name, value in weights.items():
        try:
            weight_arrays[name] = np.broadcast_to(value, leading_shape)
        except ValueError as exc:
            raise ValueError(
                f"{name} shape {value.shape} cannot broadcast to component leading shape "
                f"{leading_shape}"
            ) from exc
    stacked_weights = np.stack(list(weight_arrays.values()), axis=-1)
    if np.any((stacked_weights < 0.0) | (stacked_weights > 1.0)):
        raise ValueError("Final V mixture weights must lie in [0,1]")
    weight_sum = np.sum(stacked_weights, axis=-1)
    if not np.all(np.abs(weight_sum - 1.0) <= tolerance):
        raise ValueError("Final V mixture weights must sum to one within probability_tolerance")

    final = (
        weight_arrays["government_mixture_weight"][..., np.newaxis] * pg
        + weight_arrays["selfish_mixture_weight"][..., np.newaxis] * ps
        + weight_arrays["panic_mixture_weight"][..., np.newaxis] * pp
    )
    if not np.all(np.isfinite(final)) or np.any(final < 0.0):
        raise ValueError("Final shelter probability vector must be finite and non-negative")
    final_sum = np.sum(final, axis=-1)
    if not np.all(np.abs(final_sum - 1.0) <= tolerance):
        raise ValueError(
            "Final shelter probability vector must sum to one within probability_tolerance; "
            f"maximum error={float(np.max(np.abs(final_sum - 1.0)))}"
        )
    return final


def sample_truncated_normal(
    n: int,
    mean: float,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a normal distribution conditioned on ``[0,1]`` using ``rng``."""
    if not isinstance(n, (int, np.integer)) or int(n) < 0:
        raise ValueError(f"n must be a non-negative integer; got {n!r}")
    mean_value = float(_array("panic_rate_mean", mean))
    std_value = float(_array("panic_rate_std", std))
    if not 0.0 <= mean_value <= 1.0:
        raise ValueError("panic_rate_mean must lie in [0,1]")
    if std_value <= 0.0:
        raise ValueError("panic_rate_std must be finite and > 0")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be the configured numpy.random.Generator")
    a = (0.0 - mean_value) / std_value
    b = (1.0 - mean_value) / std_value
    samples = truncnorm.rvs(
        a,
        b,
        loc=mean_value,
        scale=std_value,
        size=int(n),
        random_state=rng,
    )
    samples = np.asarray(samples, dtype=float)
    if not np.all(np.isfinite(samples)) or np.any((samples < 0.0) | (samples > 1.0)):
        raise RuntimeError("Truncated-normal sampler produced an invalid panic rate")
    return samples


def validate_softmax_c1_config(config: Mapping[str, Any]) -> None:
    """Validate active new-model semantics before Stage 4 performs any work."""
    legacy = [name for name in ("theta_2", "beta_2") if name in config]
    if legacy:
        raise ValueError(
            "softmax_c1_v1 analytically fixes theta_2 and beta_2; remove legacy fields "
            f"{legacy} or use an explicitly versioned legacy configuration"
        )
    if config.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Stage 4 config must declare model_version: {MODEL_VERSION}; unversioned and "
            "legacy direct-weight configurations are incompatible"
        )
    required = (
        "epsilon",
        "c",
        "q",
        "probability_tolerance",
        "omega",
        "beta_t",
        "beta_a",
        "panic_rate_distribution",
        "panic_rate_mean",
        "panic_rate_std",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Stage 4 config is missing required fields: {missing}")
    _validated_curve_inputs(epsilon=config["epsilon"], c=config["c"], q=config["q"])
    epsilon = float(_array("epsilon", config["epsilon"]))
    m = 1.0 - float(_array("q", config["q"]))
    if not 0.0 < epsilon < m:
        raise ValueError(
            "Active softmax_c1_v1 configuration preserves the approved stricter constraint "
            "0 < epsilon < M, where M=1-q"
        )
    tolerance = float(_array("probability_tolerance", config["probability_tolerance"]))
    if tolerance <= 0.0:
        raise ValueError("probability_tolerance must be finite and > 0")
    omega = float(_array("omega", config["omega"]))
    if not 0.0 <= omega <= 1.0:
        raise ValueError("omega must lie in [0,1]")
    for name in ("beta_t", "beta_a"):
        if float(_array(name, config[name])) <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    if config["panic_rate_distribution"] != "truncated_normal":
        raise ValueError(
            "softmax_c1_v1 requires panic_rate_distribution: truncated_normal; "
            "gaussian_clipped is a legacy mode and is not interpreted as truncated normal"
        )
    mean = float(_array("panic_rate_mean", config["panic_rate_mean"]))
    std = float(_array("panic_rate_std", config["panic_rate_std"]))
    if not 0.0 <= mean <= 1.0:
        raise ValueError("panic_rate_mean must lie in [0,1]")
    if std <= 0.0:
        raise ValueError("panic_rate_std must be finite and > 0")
