"""Process setup required before importing heavyweight estimator backends."""

from __future__ import annotations

import importlib.util
import os


def _sole_default_target(default: str | None, available: list[str]) -> str | None:
    """Return the default only when it is the distribution's sole target."""
    if len(available) == 1 and default == available[0]:
        return default
    return None


def configure_rocm_sdk_target() -> bool:
    """Avoid the broken Windows ``offload-arch`` launcher when unambiguous.

    Current ROCm SDK Windows wheels discover the GPU target by invoking an
    ``offload-arch.exe`` console launcher. That launcher splits its installed
    executable path when the Python prefix contains spaces. If the distribution
    contains exactly one target family and declares it as the default, hardware
    discovery cannot choose anything else, so setting the documented
    ``ROCM_SDK_TARGET_FAMILY`` override is equivalent and avoids the bad probe.

    Returns ``True`` when this function supplied the override.
    """
    if os.name != "nt" or os.environ.get("ROCM_SDK_TARGET_FAMILY"):
        return False
    if importlib.util.find_spec("rocm_sdk") is None:
        return False

    from rocm_sdk import _dist_info

    target = _sole_default_target(
        getattr(_dist_info, "DEFAULT_TARGET_FAMILY", None),
        list(getattr(_dist_info, "AVAILABLE_TARGET_FAMILIES", ())),
    )
    if target is None:
        return False
    os.environ["ROCM_SDK_TARGET_FAMILY"] = target
    return True

