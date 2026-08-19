from leaf import get_model_urls_for_config, download_models_for_config
from leaf import models
from leaf.visualization import CanopyVisualizer
from leaf.visualization import FlattenedVisualizer, Path
from leaf.inference import Predictor

# Example run for flattened leaves
# intialize predictor
pred = Predictor(config_name='canopy_portrait_2')

# from $SCRATCH to reduce I/O limitations on the server
# BASE_DIR = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")
BASE_DIR = Path("/agroscope/Data-Work_CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")

dirs_to_process = [d for d in Path(BASE_DIR).iterdir() if d.is_dir()]

for d in dirs_to_process:
    pred.predict(
        images_src=d, 
        export_dst=Path(str(d).replace("B_Data", "E_Work") / "predictions")
    )