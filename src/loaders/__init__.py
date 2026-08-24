"""Dataset loaders for the SCADA one-class benchmark."""
from .batadal import build_batadal
from .gas_pipeline import build_gas_pipeline
from .hai import build_hai

BUILDERS = {
    "batadal": build_batadal,
    "gas_pipeline": build_gas_pipeline,
    "gas": build_gas_pipeline,
    "hai": build_hai,
}


def build_dataset(name: str, **kwargs):
    key = name.lower().replace("-", "_")
    if key not in BUILDERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(set(BUILDERS))}")
    return BUILDERS[key](**kwargs)


__all__ = ["build_batadal", "build_gas_pipeline", "build_hai", "build_dataset", "BUILDERS"]
