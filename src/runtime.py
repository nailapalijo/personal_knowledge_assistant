import os


def get_runtime_mode(platform=None):
    """Return the configured runtime mode, defaulting to Cloud-safe on Linux."""
    mode = os.environ.get("PKA_MODE")
    if mode is None:
        mode = "local" if (platform or os.name) == "nt" else "cloud"

    mode = mode.strip().lower()
    if mode not in {"cloud", "local"}:
        raise ValueError("PKA_MODE must be either 'cloud' or 'local'.")
    return mode
