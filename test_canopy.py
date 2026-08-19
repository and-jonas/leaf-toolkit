from leaf import get_model_urls_for_config, download_models_for_config
from leaf import models
from leaf.visualization import CanopyVisualizer
from leaf.visualization import Path
from leaf.inference import Predictor

# # pre-download models, default canopy_portrait
# downloaded = download_models_for_config()

# from $SCRATCH to reduce I/O limitations on the server
<<<<<<< HEAD
# dir_to_process = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")
ROOT_DIR = Path("/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")

# list all directories to process
dirs_to_process = [d for d in Path(ROOT_DIR).iterdir() if d.is_dir()]

# # predict
# pred = Predictor(config_name='canopy_portrait_2')
# for d in dirs_to_process:
#     pred.predict(
#         images_src=d, 
#         export_dst= Path(str(d).replace("B_Data", "E_Work")) / "predictions"
#     )

# visualize
for d in dirs_to_process:
    print(d)
    vis = CanopyVisualizer(
        vis_all=True,
        src_root=Path(str(d).replace("B_Data", "E_Work")) / "predictions",
        rgb_root=d,
        export_root=Path(str(d).replace("B_Data", "E_Work")) / "predictions")
    vis.visualize()
=======
# BASE_DIR = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")
BASE_DIR = Path("/agroscope/Data-Work_CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")

dirs_to_process = [d for d in Path(BASE_DIR).iterdir() if d.is_dir()]

for d in dirs_to_process:
    pred.predict(
        images_src=d, 
        export_dst=Path(str(d).replace("B_Data", "E_Work") / "predictions")
    )
>>>>>>> 1e27bf666a0fb1a185a0c9b8684b06079c9e774c
