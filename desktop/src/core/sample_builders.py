"""Helpers that translate monitor outputs into per-sample DataSample fields.

Shared by both the user app's monitoring loop and the dev app's recorder so
the two stay in lockstep — fields produced here go straight into a
:class:`DataSample`.
"""


def flatten_hw_latest(latest: dict, timestamp: float) -> dict:
    """Merge :meth:`HardwareMonitor.get_latest` into a flat per-sample dict.

    Returns a fixed-shape snapshot (missing fields → ``None``); never raises
    when GPU/disk readings are absent.
    """
    cpu = latest.get("cpu") or {}
    ram = latest.get("ram") or {}
    gpu = latest.get("gpu")
    disk = latest.get("disk") or {}
    return {
        "timestamp": timestamp,
        "cpu_percent": cpu.get("percent"),
        "cpu_freq_ghz": cpu.get("freq_ghz"),
        "ram_percent": ram.get("percent"),
        "ram_used_gb": ram.get("used_gb"),
        "gpu_present": 1 if gpu else 0,
        "gpu_load_percent": gpu.get("load_percent") if gpu else None,
        "gpu_memory_used_gb": gpu.get("memory_used_gb") if gpu else None,
        "gpu_temperature_c": gpu.get("temperature_c") if gpu else None,
        "disk_read_bps": disk.get("read_bps"),
        "disk_write_bps": disk.get("write_bps"),
    }


def build_input_since_last(input_monitor) -> dict:
    """Drain input aggregates and flicks for the current sample tick.

    Falls back gracefully when the supplied monitor predates the new
    drain methods — the legacy ``get_and_reset_count`` is still consulted
    so a basic total count is always populated.
    """
    if hasattr(input_monitor, "get_and_reset_input_aggregates"):
        agg = input_monitor.get_and_reset_input_aggregates()
    else:
        total = (
            input_monitor.get_and_reset_count()
            if hasattr(input_monitor, "get_and_reset_count")
            else 0
        )
        agg = {
            "key_press_count": 0,
            "mouse_click_count": 0,
            "mouse_scroll_count": 0,
            "mouse_move_count": 0,
            "total_count": total,
            "gaming_keys": {
                "w": 0, "a": 0, "s": 0, "d": 0,
                "space": 0, "shift": 0, "left": 0, "right": 0,
            },
        }

    flicks = (
        input_monitor.get_and_reset_flicks()
        if hasattr(input_monitor, "get_and_reset_flicks")
        else []
    )

    if flicks:
        n = len(flicks)
        mags = [(dx * dx + dy * dy) ** 0.5 for dx, dy in flicks]
        agg["flick_count"] = n
        agg["flick_mag_mean"] = sum(mags) / n
        agg["flick_mag_max"] = max(mags)
        agg["flick_dx_mean"] = sum(dx for dx, _ in flicks) / n
        agg["flick_dy_mean"] = sum(dy for _, dy in flicks) / n
    else:
        agg["flick_count"] = 0
        agg["flick_mag_mean"] = 0.0
        agg["flick_mag_max"] = 0.0
        agg["flick_dx_mean"] = 0.0
        agg["flick_dy_mean"] = 0.0
    return agg
