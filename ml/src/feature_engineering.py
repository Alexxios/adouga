"""Convert a raw DataSample dict into a fixed-size feature vector for the tabular branch.

The feature vector has ``TABULAR_DIM`` floats and is built from hardware histories,
input statistics, flick vectors, input sequences and key-press heatmaps.

Each variable-length history is reduced to five summary statistics per numeric
field: *mean, std, min, max, last*.  Empty histories produce zeros.
"""

import logging
import math

logger = logging.getLogger(__name__)

TABULAR_DIM = 102

# Keys we watch for in the 3-minute heatmap window.
_GAMING_KEYS = ("w", "a", "s", "d", "space", "shift", "left", "right")

# Ordered heatmap window labels (must match InputMonitor output).
_HEATMAP_LABELS = ("1s", "5s", "15s", "30s", "1m", "3m")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float:
    """Coerce *val* to float, treating ``None`` as 0.0."""
    if val is None:
        return 0.0
    return float(val)


def _stats(values: list[float]) -> list[float]:
    """Return [mean, std, min, max, last] for a non-empty list, or 5 zeros."""
    if not values:
        return [0.0] * 5
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return [mean, math.sqrt(var), min(values), max(values), values[-1]]


def _history_stats(entries: list[dict], fields: list[str]) -> list[float]:
    """Extract per-field summary statistics from a list of history dicts.

    Returns ``5 * len(fields)`` floats.
    """
    result: list[float] = []
    for field in fields:
        vals = [_safe_float(e.get(field)) for e in entries]
        result.extend(_stats(vals))
    return result


def _flick_stats(flick_vectors: list) -> list[float]:
    """Return 8 features from flick vectors: count, mag stats, dx/dy stats."""
    if not flick_vectors:
        return [0.0] * 8
    dxs = [float(f[0]) for f in flick_vectors]
    dys = [float(f[1]) for f in flick_vectors]
    mags = [math.sqrt(dx * dx + dy * dy) for dx, dy in zip(dxs, dys)]
    n = len(mags)
    mean_mag = sum(mags) / n
    var_mag = sum((m - mean_mag) ** 2 for m in mags) / n
    mean_dx = sum(dxs) / n
    mean_dy = sum(dys) / n
    var_dx = sum((d - mean_dx) ** 2 for d in dxs) / n
    var_dy = sum((d - mean_dy) ** 2 for d in dys) / n
    return [
        float(n),
        mean_mag,
        math.sqrt(var_mag),
        max(mags),
        mean_dx,
        mean_dy,
        math.sqrt(var_dx),
        math.sqrt(var_dy),
    ]


def _sequence_stats(input_sequence: list[dict]) -> list[float]:
    """Return 5 features: per-type counts, total, events/sec."""
    if not input_sequence:
        return [0.0] * 5
    key_count = sum(1 for e in input_sequence if e.get("type") == "key_press")
    click_count = sum(1 for e in input_sequence if e.get("type") == "mouse_click")
    scroll_count = sum(1 for e in input_sequence if e.get("type") == "mouse_scroll")
    total = len(input_sequence)
    timestamps = [e["timestamp"] for e in input_sequence]
    duration = max(timestamps) - min(timestamps)
    eps = total / duration if duration > 0 else 0.0
    return [float(key_count), float(click_count), float(scroll_count), float(total), eps]


def _shannon_entropy(counts: dict[str, int]) -> float:
    """Compute Shannon entropy (bits) of a {key: count} dict."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def _heatmap_stats(key_heatmaps: dict) -> list[float]:
    """Return 24 features (4 per window) + 8 gaming-key counts = 32 total."""
    result: list[float] = []
    for label in _HEATMAP_LABELS:
        window = key_heatmaps.get(label, {})
        if not window:
            result.extend([0.0, 0.0, 0.0, 0.0])
        else:
            total = float(sum(window.values()))
            unique = float(len(window))
            mx = float(max(window.values()))
            ent = _shannon_entropy(window)
            result.extend([total, unique, mx, ent])
    # Gaming-key counts from the 3m window
    window_3m = key_heatmaps.get("3m", {})
    for key in _GAMING_KEYS:
        result.append(float(window_3m.get(key, 0)))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_tabular_features(sample: dict) -> list[float]:
    """Convert a single sample dict to a flat list of ``TABULAR_DIM`` floats.

    Parameters
    ----------
    sample:
        A dict matching the JSON output of ``DataSample.to_dict()`` — the
        format stored in ``samples.jsonl`` inside exported ZIP archives.

    Returns
    -------
    list[float]
        Feature vector of length ``TABULAR_DIM``.
    """
    features: list[float] = []

    # CPU (10)
    features.extend(_history_stats(
        sample.get("cpu_history", []),
        ["percent", "freq_ghz"],
    ))

    # RAM (15)
    features.extend(_history_stats(
        sample.get("ram_history", []),
        ["percent", "used_gb", "total_gb"],
    ))

    # GPU (20 + 1 flag)
    gpu_hist = sample.get("gpu_history", [])
    features.extend(_history_stats(
        gpu_hist,
        ["load_percent", "memory_used_gb", "memory_total_gb", "temperature_c"],
    ))
    features.append(1.0 if gpu_hist else 0.0)

    # Disk (10)
    features.extend(_history_stats(
        sample.get("disk_history", []),
        ["read_bps", "write_bps"],
    ))

    # Input count (1)
    features.append(float(sample.get("input_count", 0)))

    # Flick vectors (8)
    features.extend(_flick_stats(sample.get("flick_vectors", [])))

    # Input sequence (5)
    features.extend(_sequence_stats(sample.get("input_sequence", [])))

    # Heatmaps (32)
    features.extend(_heatmap_stats(sample.get("key_heatmaps", {})))

    assert len(features) == TABULAR_DIM, (
        f"Expected {TABULAR_DIM} features, got {len(features)}"
    )
    return features
