from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_MODEL_URLS: Dict[tuple[str, str], str] = {
    # symptoms_seg
    ('symptoms_seg', 'latest'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/seg_fpn_mitb3_tkbkocy7.torchscript',
    ('symptoms_seg', 'tracking_latest'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.1/seg_fpn_mitb2_0.001_1024_seg_tracking_v1.torchscript',
    ('symptoms_seg', 'zenkl_et_al_2025b'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/seg_fpn_mitb3_tkbkocy7.torchscript',
    ('symptoms_seg', 'anderegg_et_al_2025'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.1/seg_fpn_mitb2_0.001_1024_seg_tracking_v1.torchscript',
    ('symptoms_seg', 'zenkl_et_al_2025a'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v0.3.0/fpn_mitb1_v4.torchscript',
    ('symptoms_seg', 'anderegg_et_al_2024'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v0.3.1/fpn_mitb1_dsnv6.torchscript',

    # organs
    ('organs', 'latest'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/org_fpn_mitb2_b6oo1xel.torchscript',
    ('organs', 'zenkl_et_al_2025b'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/org_fpn_mitb2_b6oo1xel.torchscript',

    # symptoms_det
    ('symptoms_det', 'latest'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/yolo11l-pose_t11z7ymj.pt',
    ('symptoms_det', 'tracking_latest'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.1/yolo11m-pose-key_tracking_v2.pt',
    ('symptoms_det', 'zenkl_et_al_2025b'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/yolo11l-pose_t11z7ymj.pt',
    ('symptoms_det', 'zenkl_et_al_2025a'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v0.3.0/yolov8m-pose-v4.4.pt',
    ('symptoms_det', 'anderegg_et_al_2025'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.1/yolo11m-pose-key_tracking_v2.pt',
    ('symptoms_det', 'anderegg_et_al_2024'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v0.3.1/yolov8m-pose_v7.pt',

    # focus
    ('focus', 'latest'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/4096x6144_1x3x1036x1540_DepthAnythingV2_vits.pt',
    ('focus', '6144x4096'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/6144x4096_1x3x1540x1036_DepthAnythingV2_vits.pt',
    ('focus', '4096x6144'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/4096x6144_1x3x1036x1540_DepthAnythingV2_vits.pt',
    ('focus', '4096x4096'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/4096x4096_1x3x1036x1036_DepthAnythingV2_vits.pt',
    ('focus', '2048x2048'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/2048x2048_1x3x518x518_DepthAnythingV2_vits.pt',
    ('focus', '1024x1024'): 'https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/1024x1024_1x3x266x266_DepthAnythingV2_vits.pt',
}

_MODULE_NAMES = {'symptoms_det', 'symptoms_seg', 'organs', 'focus'}


def get_config(config_name: str = 'canopy_portrait', config_path: str = 'config') -> dict:
    """Load a YAML config from the package config directory."""
    config_dir = Path(__file__).resolve().parent / config_path
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)

    return OmegaConf.to_container(cfg, resolve=True)


def download_file(url: str, root: str = 'models') -> str:
    """Download a file from a URL into a root directory, skipping if already present."""
    filename = Path(url).name
    destination = Path(root) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        logging.info(f"File already exists: {destination}")
        return str(destination)

    logging.info(f"Downloading {url} to {destination}")
    urllib.request.urlretrieve(url, str(destination))
    logging.info(f"Downloaded {destination}")
    return str(destination)


def get_model_urls_for_config(
    config_name: str = 'canopy_portrait',
    config_path: str = 'config',
    modules: Optional[List[str]] = None,
    download_disabled_modules: bool = False,
) -> Dict[str, str]:
    """Return the GitHub download URLs for models referenced by a config."""
    config = get_config(config_name=config_name, config_path=config_path)
    module_params = config.get('module_params', {}) or {}

    if modules is not None:
        selected_modules = [m for m in modules if m in _MODULE_NAMES]
        if not selected_modules:
            raise ValueError(f"No valid modules selected. Supported modules: {_MODULE_NAMES}")
    elif module_params:
        selected_modules = [m for m, enabled in module_params.items() if enabled and m in _MODULE_NAMES]
    else:
        selected_modules = [m for m in _MODULE_NAMES if config.get(f'{m}_params')]

    urls: Dict[str, str] = {}
    for module_name in selected_modules:
        enabled = module_params.get(module_name, True)
        if not enabled and not download_disabled_modules:
            continue

        params = config.get(f'{module_name}_params', {}) or {}
        model_name = params.get('model_name')
        if model_name is None:
            logging.warning("Skipping %s because no model_name is configured.", module_name)
            continue

        url = _MODEL_URLS.get((module_name, model_name))
        if url is None:
            raise ValueError(
                f"No download URL found for module '{module_name}' with model_name '{model_name}'"
            )

        urls[module_name] = url

    return urls


def download_models_for_config(
    config_name: str = 'canopy_portrait',
    config_path: str = 'config',
    model_root: str = 'models',
    modules: Optional[List[str]] = None,
    download_disabled_modules: bool = False,
) -> Dict[str, str]:
    """Download all model files referenced by a configuration."""
    urls = get_model_urls_for_config(
        config_name=config_name,
        config_path=config_path,
        modules=modules,
        download_disabled_modules=download_disabled_modules,
    )

    downloaded: Dict[str, str] = {}
    for module_name, url in urls.items():
        downloaded[module_name] = download_file(url, root=model_root)

    return downloaded
