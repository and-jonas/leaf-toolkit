from leaf import get_model_urls_for_config, download_models_for_config
from leaf import models
from leaf.visualization import CanopyVisualizer
from leaf.metrics import canopy_evaluation_wrapper
from leaf.inference import Predictor
import logging

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(processName)s | %(levelname)s | %(message)s'
)

# pre-download models
urls = get_model_urls_for_config(config_name='flattened_leaves', config_path='config')
downloaded = download_models_for_config()
downloaded = download_models_for_config(config_name="canopy_landscape")
downloaded = download_models_for_config(config_name="flattened_leaves")

# # test models using the canopy_portrait config
# models.test()

# intialize predictor
pred = Predictor(config_name='flattened_leaves',
                 symptoms_seg_params={'model_name': 'tracking_latest',
                                      'use_gpu': False},
                 symptoms_det_params={'model_name': 'tracking_latest',
                                      'use_gpu': False,
                                      'keypoints_thresh': 0.18}
)


# list directories to process
dir_to_process = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/WW40/1260/test")

# predict
pred.predict(
    images_src=f'{dir_to_process}/inference_crops', 
    export_dst=f'{dir_to_process}/predictions',
    devices=['cpu', 'cpu'])

# # create visualizations 
# vis = CanopyVisualizer(
#     vis_all=True,
#     src_root='test/export',
#     rgb_root='test/images',
#     export_root='export',
#     )
# vis.visualize()

# # extract metrics
# canopy_evaluation_wrapper(root_folder='export', results_path='export/canopy_results.csv')