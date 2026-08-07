from leaf import get_model_urls_for_config, download_models_for_config
from leaf import models
from leaf.visualization import CanopyVisualizer
from leaf.visualization import FlattenedVisualizer, Path
from leaf.inference import Predictor


# urls = get_model_urls_for_config(config_name='flattened_leaves', config_path='config')
# print(urls)

# # pre-download models, default canopy_portrait
# downloaded = download_models_for_config()
# # pre-download models, canopy_landscape
# downloaded = download_models_for_config(
#     config_name="canopy_landscape"
# )
# # pre-download models, flattened_leaves
# downloaded = download_models_for_config(
#     config_name="flattened_leaves"
# )

# models.test()

# vis = CanopyVisualizer(
#     vis_all=True,
#     src_root='test/export',
#     rgb_root='test/images',
#     export_root='export',
#     )
# vis.visualize()

# Example run for flattened leaves
# intialize predictor
pred = Predictor(config_name='flattened_leaves',
                 symptoms_seg_params={'model_name': 'tracking_latest',
                                      'use_gpu': False},
                 symptoms_det_params={'model_name': 'tracking_latest',
                                      'use_gpu': False,
                                      'keypoints_thresh': 0.18}
)

# from $SCRATCH to reduce I/O limitations on the server
dir_to_process = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/06_WW40/LeafImages/1260/20260610")

# # predict
# pred.predict(
#     images_src=dir_to_process / "subset", 
#     export_dst=dir_to_process / "predictions"
# )

# visualize
vis = FlattenedVisualizer(
    src_root=dir_to_process / "predictions", 
    rgb_root=dir_to_process / "subset", 
    export_root=dir_to_process / "predictions"
)
vis.visualize()