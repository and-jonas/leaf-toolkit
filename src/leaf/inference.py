import logging
import torch
from typing import Union

from hydra import compose, initialize
from omegaconf import OmegaConf
from collections import defaultdict

from leaf.models import SymptomsDetection, SymptomsSegmentation, OrgansSegmentation, FocusSegmentation
from leaf.preprocessing import Preprocessor
from pathlib import Path
import cv2

import matplotlib.pyplot as plt


class Predictor:
    """
    This class is used to unify the complete pipeline into a single object with a simple predict method. 
    It provides capability to use a yaml file configuration for repeatability. Furthermore, it allows for 
    passing all the arguments for cofigring the individual building blocks of the pipeline. If a new block 
    """

    def __init__(self,
                 config_path: str = "config", config_name: str = "canopy_portrait",
                 preprocessing_params: Union[dict, None] = None,
                 symptoms_det_params: Union[dict, None] = None,
                 symptoms_seg_params: Union[dict, None] = None,
                 organs_params: Union[dict, None] = None,
                 focus_params: Union[dict, None] = None,
                 module_params:  Union[dict, None] = None
                 ) -> None:
        """
        Constructor of the predictor object. For possible parameters to configure see either the individual
        models directly or see the configuration yaml files in the config folder. 

        Args:
            config_path (str, optional): relative path from the location of this file to a config directory. 
                Defaults to "config".
            config_name (str, optional): name of a config within the config_path directory. 
                New configurations can be added. Defaults to "canopy_portrait".
            symptoms_det_params (Union[dict, None], optional): An optional dictionary which directly passes 
                the contents as **kwargs to symptoms detection model. It overrides the parameters from the 
                configuration file. Defaults to None.
            symptoms_seg_params (Union[dict, None], optional): An optional dictionary which directly passes 
                the contents as **kwargs to symptoms segmentation model. It overrides the parameters from the 
                configuration file. Defaults to None.
            organs_params (Union[dict, None], optional): An optional dictionary which directly passes 
                the contents as **kwargs to organs segmentation model. It overrides the parameters from the 
                configuration file. Defaults to None.
            focus_params (Union[dict, None], optional): An optional dictionary which directly passes 
                the contents as **kwargs to focus estimation model. It overrides the parameters from the 
                configuration file. Defaults to None.
            module_params (Union[dict, None], optional): An optional dictionary which controls which parts 
                of the pipeline are executed. It overrides the parameters from the configuration file. 
                Defaults to None.
        """
        
        
        # load base config
        with initialize(version_base=None, config_path=config_path):
            cfg = compose(config_name=config_name)
            config = OmegaConf.to_container(cfg, resolve=True)

        self.preprocessing_params = config.get('preprocessing_params', None)
        self.module_params = config.get('module_params', None)
        self.symptoms_det_params = config.get('symptoms_det_params', None)
        self.symptoms_seg_params = config.get('symptoms_seg_params', None)
        self.organs_params =  config.get('organs_params', None)
        self.focus_params =  config.get('focus_params', None)

        # override with user params
        if preprocessing_params is not None:
            self.preprocessing_params.update(preprocessing_params)
        if module_params is not None:
            self.module_params.update(module_params)
        if symptoms_det_params is not None:
            self.symptoms_det_params.update(symptoms_det_params)
        if symptoms_seg_params is not None:
            self.symptoms_seg_params.update(symptoms_seg_params)
        if organs_params is not None:
            self.organs_params.update(organs_params)
        if focus_params is not None:
            self.focus_params.update(focus_params)
        
    def predict(self, images_src: str, export_dst: str) -> None:
        """
        This method provides a simple interface to predict on images from a specified folder and 
        save the results to a specified location.

        Args:
            images_src (str): Path to location of images.
            export_dst (str): Path where the results should be saved.
        """

        logging.info("Predicting ...")
        logging.info("Emptying CUDA cache")
        torch.cuda.empty_cache()
        

        # instantiate single preprocessor (rotate/crop) and models
        # allow per-config image crop parameters under `module_params['preprocessing']`
        preproc_cfg = self.preprocessing_params if self.preprocessing_params else {}
        crop_sz = tuple(preproc_cfg.get('crop_sz')) if preproc_cfg.get('crop_sz') else None
        crop_offsets = tuple(preproc_cfg.get('crop_offsets')) if preproc_cfg.get('crop_offsets') else None
        preproc = Preprocessor()
        if crop_sz is not None:
            preproc.crop_sz = crop_sz
        if crop_offsets is not None:
            preproc.crop_offsets = crop_offsets

        models = {}
        if self.module_params.get('symptoms_det'):
            models['symptoms_det'] = SymptomsDetection(
                **self.symptoms_det_params,
                export_pattern_pred=f'{export_dst}/symptoms_det/pred',
            )
        if self.module_params.get('symptoms_seg'):
            models['symptoms_seg'] = SymptomsSegmentation(
                **self.symptoms_seg_params,
                export_pattern_pred=f'{export_dst}/symptoms_seg/pred',
            )
        if self.module_params.get('organs'):
            models['organs'] = OrgansSegmentation(
                **self.organs_params,
                export_pattern_pred=f'{export_dst}/organs/pred',
            )
        if self.module_params.get('focus'):
            models['focus'] = FocusSegmentation(
                **self.focus_params,
                export_pattern_pred=f'{export_dst}/focus/pred',
            )

        # gather files
        src_path = Path(images_src)
        search_pattern = ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG']
        if src_path.is_dir():
            raw_files = [file for ext in search_pattern for file in src_path.rglob(ext)]
            dedup = {}
            for file in raw_files:
                normalized = str(file.resolve()).casefold()
                dedup[normalized] = file.resolve()
            files = sorted(dedup.values())
        elif src_path.is_file():
            files = [src_path.resolve()]
        else:
            logging.error(f"images_src not found: {images_src}")
            return

        for file in files:
            logging.debug(f"Processing {file}")
            img = cv2.imread(str(file))
            if img is None:
                logging.error(f"Failed to read {file}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # run module-agnostic preprocessing once
            img_proc = preproc.preprocess_image(img)

            # pass processed image to each model (models decide their own per-image and per-patch preprocessing)
            for name, model in models.items():
                model.predict_from_array(img_proc, file)
                              
        logging.info("Predicting finished")
