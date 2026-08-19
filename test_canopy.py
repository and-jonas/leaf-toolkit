from leaf import get_model_urls_for_config, download_models_for_config
from leaf import models
from leaf.visualization import CanopyVisualizer
from leaf.visualization import Path
from leaf.inference import Predictor

# # pre-download models, default canopy_portrait
# downloaded = download_models_for_config()

# from $SCRATCH to reduce I/O limitations on the server
# ROOT_DIR = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")
# ROOT_DIR = Path("/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")
ROOT_DIR = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260622_Uitikon")
# ROOT_DIR = Path("/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260622_Uitikon")

# list all directories to process
dirs_to_process = [
    d
    for camera_dir in Path(ROOT_DIR).iterdir()
    if camera_dir.is_dir() and camera_dir.name.startswith("Camera")
    for d in camera_dir.iterdir()
    if d.is_dir()
]

# predict
pred = Predictor(config_name='canopy_portrait_2')
for d in dirs_to_process:
    pred.predict(
        images_src=d, 
        export_dst= Path(str(d).replace("B_Data", "E_Work")) / "predictions"
    )

# visualize
for d in dirs_to_process:
    print(d)
    vis = CanopyVisualizer(
        config_path="config", config_name='canopy_portrait_2',
        vis_all=True,
        src_root=Path(str(d).replace("B_Data", "E_Work")) / "predictions",
        rgb_root=d,
        export_root=Path(str(d).replace("B_Data", "E_Work")) / "predictions")
    vis.visualize(parallel=True)
