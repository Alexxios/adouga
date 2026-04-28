"""Convert a per-sample DataSample dict into a fixed-size feature vector.

Layout (deterministic — index-aligned with the model's tabular branch):

  [0:50)   HW recent — 5 slots × 10 fields, oldest first; missing slots are
           zeroed and ``gpu_present`` already encodes GPU absence.
  [50:56)  Input counts — key_press / mouse_click / mouse_scroll / mouse_move
           / total / + 1 reserved slot (currently 0 to keep the dim stable).
  [56:61)  Flick stats — count, mag_mean, mag_max, dx_mean, dy_mean.
  [61:69)  Gaming key counts — w, a, s, d, space, shift, left, right.
  [69:85)  16-bucket SHA-256 indicator over the lower-cased ``app_name``.

Heavy-tailed quantities (disk bps, all input counts, flick magnitudes) are
``log1p``-transformed at extraction time so the tabular branch sees the
same scales it will see in production. Empty / missing fields → zeros.
"""

import hashlib
import math

TABULAR_DIM = 85

_HW_RECENT_DEPTH = 5
_HW_FIELDS: tuple[tuple[str, bool], ...] = (
    # (field, log1p_transform)
    ("cpu_percent",        False),
    ("cpu_freq_ghz",       False),
    ("ram_percent",        False),
    ("ram_used_gb",        False),
    ("gpu_present",        False),
    ("gpu_load_percent",   False),
    ("gpu_memory_used_gb", False),
    ("gpu_temperature_c",  False),
    ("disk_read_bps",      True),
    ("disk_write_bps",     True),
)
assert len(_HW_FIELDS) == 10  # noqa: PLR2004 — schema invariant

_GAMING_KEYS: tuple[str, ...] = (
    "w", "a", "s", "d", "space", "shift", "left", "right",
)
_APP_HASH_BUCKETS = 16


def _safe_float(val) -> float:
    """Coerce *val* to float; ``None`` and unparseable values become 0.0."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _hw_slot(entry: dict) -> list[float]:
    """Encode a single HW snapshot dict into 10 floats (per ``_HW_FIELDS``)."""
    out: list[float] = []
    for field, use_log1p in _HW_FIELDS:
        v = _safe_float(entry.get(field))
        if use_log1p:
            v = math.log1p(max(0.0, v))
        out.append(v)
    return out


def _hw_recent_block(hw_recent: list) -> list[float]:
    """Pad/truncate to exactly ``_HW_RECENT_DEPTH`` slots, oldest first."""
    entries = hw_recent or []
    if len(entries) > _HW_RECENT_DEPTH:
        entries = entries[-_HW_RECENT_DEPTH:]
    out: list[float] = []
    pad = _HW_RECENT_DEPTH - len(entries)
    for _ in range(pad):
        out.extend([0.0] * len(_HW_FIELDS))
    for entry in entries:
        out.extend(_hw_slot(entry if isinstance(entry, dict) else {}))
    return out


def _input_block(input_since_last: dict) -> list[float]:
    """6 input counts (log1p-transformed) — last slot reserved for layout stability."""
    src = input_since_last or {}
    counts = [
        _safe_float(src.get("key_press_count")),
        _safe_float(src.get("mouse_click_count")),
        _safe_float(src.get("mouse_scroll_count")),
        _safe_float(src.get("mouse_move_count")),
        _safe_float(src.get("total_count")),
    ]
    return [math.log1p(max(0.0, v)) for v in counts] + [0.0]


def _flick_block(input_since_last: dict) -> list[float]:
    """5 flick stats; counts/magnitudes pass through log1p."""
    src = input_since_last or {}
    return [
        math.log1p(max(0.0, _safe_float(src.get("flick_count")))),
        math.log1p(max(0.0, _safe_float(src.get("flick_mag_mean")))),
        math.log1p(max(0.0, _safe_float(src.get("flick_mag_max")))),
        _safe_float(src.get("flick_dx_mean")),
        _safe_float(src.get("flick_dy_mean")),
    ]


def _gaming_keys_block(input_since_last: dict) -> list[float]:
    """8 per-key press counts (log1p), in the canonical key order."""
    keys = (input_since_last or {}).get("gaming_keys") or {}
    return [
        math.log1p(max(0.0, _safe_float(keys.get(k))))
        for k in _GAMING_KEYS
    ]


def _app_hash_block(app_name: str) -> list[float]:
    """One-hot 16-bucket SHA-256 indicator over lower-cased app name."""
    out = [0.0] * _APP_HASH_BUCKETS
    if not app_name:
        return out
    digest = hashlib.sha256(app_name.strip().lower().encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % _APP_HASH_BUCKETS
    out[bucket] = 1.0
    return out


def extract_tabular_features(sample: dict) -> list[float]:
    """Convert one sample dict to a flat list of ``TABULAR_DIM`` floats.

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
    features.extend(_hw_recent_block(sample.get("hw_recent", [])))
    features.extend(_input_block(sample.get("input_since_last", {})))
    features.extend(_flick_block(sample.get("input_since_last", {})))
    features.extend(_gaming_keys_block(sample.get("input_since_last", {})))
    features.extend(_app_hash_block(sample.get("app_name", "")))

    assert len(features) == TABULAR_DIM, (
        f"Expected {TABULAR_DIM} features, got {len(features)}"
    )
    return features
