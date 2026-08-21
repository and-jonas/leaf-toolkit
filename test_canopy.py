from leaf import get_model_urls_for_config, download_models_for_config
from leaf import models
from leaf.visualization import CanopyVisualizer
from leaf.inference import Predictor
from pathlib import Path
import re

# # pre-download models, default canopy_portrait
# downloaded = download_models_for_config()

# from $SCRATCH to reduce I/O limitations on the server
# ROOT_DIR = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")
# ROOT_DIR = Path("/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")
# ROOT_DIR = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260622_Uitikon")
# ROOT_DIR = Path("/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260622_Uitikon")

# determine which root path to use based on existence of local or server path
ROOT_LOCAL = Path(
    "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/"
    "Jonas_Anderegg_Files/B_Data/03_PreDiMix"
)
ROOT_SERVER = Path(
    "/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/"
    "Jonas_Anderegg_Files/B_Data/03_PreDiMix"
)
if ROOT_LOCAL.exists():
    ROOT = ROOT_LOCAL
elif ROOT_SERVER.exists():
    ROOT = ROOT_SERVER
else:
    raise FileNotFoundError("Could not find PreDiMix root directory.")

# list all directories in the root directory that match the date format and do not contain certain substrings
SITE_DIRS = [d for d in ROOT.iterdir() if d.is_dir()]
DATE_DIRS = [
    d
    for site_dir in SITE_DIRS
    for d in site_dir.iterdir()
    if d.is_dir() and re.match(r"^\d{8}_", d.name) and not any(x in d.name for x in ["_Leaf", "_Documentation", "_Test"])
]
CAMERA_DIRS = [
    d
    for date_dir in DATE_DIRS
    for d in date_dir.iterdir()
    if d.is_dir() and d.name.startswith("Camera")
]

# initialize predictor with the 'canopy_portrait_2' configuration
pred = Predictor(config_name='canopy_portrait_2')

# predict
for d in CAMERA_DIRS:
    pred.predict(
        images_src=d,
        export_dst=Path(str(d).replace("B_Data", "E_Work")) / "predictions"
    )

# # visualize
# for d in CAMERA_DIRS:
#     print(d)
#     vis = CanopyVisualizer(
#         config_path="config", config_name='canopy_portrait_2',
#         vis_all=True,
#         src_root=Path(str(d).replace("B_Data", "E_Work")) / "predictions",
#         rgb_root=d,
#         export_root=Path(str(d).replace("B_Data", "E_Work")) / "predictions")
#     vis.visualize(parallel=True)
