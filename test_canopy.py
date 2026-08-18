from leaf import get_model_urls_for_config, download_models_for_config
from leaf import models
from leaf.visualization import CanopyVisualizer
from leaf.visualization import FlattenedVisualizer, Path
from leaf.inference import Predictor

# Example run for flattened leaves
# intialize predictor
pred = Predictor(config_name='canopy_portrait_2')

# from $SCRATCH to reduce I/O limitations on the server
# dir_to_process = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")
dir_to_process = Path("/agroscope/Data-Work_CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260521_Uitikon_Test/Camera1")

# predict
pred.predict(
    images_src=dir_to_process, 
    export_dst=dir_to_process / "predictions"
)