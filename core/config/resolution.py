"""Single source of truth for train/inference resolution.

The value lives in `configs/resolution.yaml`; this helper is imported by both entry points
(`core/finetune/schemas/args.py` for training, `infer_nf4.py` for inference) as the default for
`--train_resolution`, so changing the YAML permeates everywhere. Internal masking does NOT read this
value — it derives the exo/ego split from tensor shapes (ego is square ⇒ ego_latent_width = latent_height)
so the mask can never disagree with the data.
"""
from pathlib import Path

import yaml

# core/config/resolution.py -> parents[2] == repo root (EgoX/) -> configs/resolution.yaml
_CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "resolution.yaml"


def get_train_resolution() -> str:
    """Return the 'FxHxW' resolution string from configs/resolution.yaml (e.g. '49x176x704')."""
    return str(yaml.safe_load(_CFG_PATH.read_text())["train_resolution"])
