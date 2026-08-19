import os
from pathlib import Path

from leaf.visualization import Path
from leaf.inference import Predictor


ROOT_DIR = Path(
    "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/"
    "Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260622_Uitikon"
)

# Find all directories directly inside Camera* directories
dirs_to_process = sorted([
    d
    for camera_dir in ROOT_DIR.iterdir()
    if camera_dir.is_dir() and camera_dir.name.startswith("Camera")
    for d in camera_dir.iterdir()
    if d.is_dir()
])

# SLURM array task ID
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# Number of array tasks
num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

# Split directories between tasks
dirs_for_this_task = dirs_to_process[task_id::num_tasks]

print(f"Task {task_id}/{num_tasks}")
print(f"Processing {len(dirs_for_this_task)} directories")

# One Predictor per process/GPU
pred = Predictor(config_name="canopy_portrait_2")

for d in dirs_for_this_task:
    print(f"Processing: {d}", flush=True)

    pred.predict(
        images_src=d,
        export_dst=Path(str(d).replace("B_Data", "E_Work")) / "predictions",
    )