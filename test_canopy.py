import os

from leaf import get_model_urls_for_config, download_models_for_config
from leaf import models
from leaf.visualization import CanopyVisualizer
from leaf.inference import Predictor
from leaf.metrics import canopy_evaluation_wrapper
from pathlib import Path
import re
from tqdm import tqdm

# # pre-download models, default canopy_portrait
# downloaded = download_models_for_config()

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

# list all plot directories in the data structure
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
PLOT_DIRS = [
    d
    for plot_dir in CAMERA_DIRS
    for d in plot_dir.iterdir()
    if d.is_dir() and re.fullmatch(r"[A-Za-z0-9]{8}", d.name)
]

print(f'found {len(PLOT_DIRS)} plot directories to process')

# split up list of plots into task lists to assign to different GPUs, if available
if "SLURM_ARRAY_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_COUNT" in os.environ:
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    PLOT_DIRS_TASK = PLOT_DIRS[task_id::num_tasks]
else:
    PLOT_DIRS_TASK = PLOT_DIRS
    task_id = 0

# initialize predictor with the 'canopy_portrait_2' configuration
pred = Predictor(config_name='canopy_portrait_temp')

# predict each plot directory, skipping those that have already been processed
for d in tqdm(
    PLOT_DIRS_TASK, 
    desc=f"Task {task_id}",
    position=task_id,
    leave=True
    ):

    export_dst = Path(str(d).replace("B_Data", "E_Work")) / "predictions"
    if export_dst.exists():
        print(f"Skipping, output already exists: {export_dst}")
        continue

    pred.predict(
        images_src=d,
        export_dst=export_dst
    )

# # get metrics for each plot directory
# for d in tqdm(
#     PLOT_DIRS_TASK, 
#     desc=f"Task {task_id}",
#     position=task_id,
#     leave=True
#     ):
#     export_dst = Path(str(d).replace("B_Data", "E_Work")) / "predictions"
#     canopy_evaluation_wrapper(root_folder=export_dst, results_path=export_dst / 'canopy_results.csv')


# # visualize
# for d in tqdm(
#     PLOT_DIRS_TASK, 
#     desc=f"Task {task_id}",
#     position=task_id,
#     leave=True
#     ):
#     vis = CanopyVisualizer(
#         config_path="config", config_name='canopy_portrait_2',
#         vis_all=True,
#         src_root=Path(str(d).replace("B_Data", "E_Work")) / "predictions",
#         rgb_root=d,
#         export_root=Path(str(d).replace("B_Data", "E_Work")) / "predictions",
#         sample_step=15
#         )
#     vis.visualize(parallel=True)
