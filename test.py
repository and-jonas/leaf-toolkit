from leaf import get_model_urls_for_config, download_models_for_config
from leaf import models
from leaf.visualization import CanopyVisualizer

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

vis = CanopyVisualizer(
    vis_all=True,
    src_root='test/export',
    rgb_root='test/images',
    export_root='export',
    )
vis.visualize()