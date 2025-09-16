import os
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import zoom
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


if __name__ == "__main__":

    data_tag = "nnUNet_data6"

    raw_path = os.environ.get("nnUNet_raw")
    preprocessed_path = os.environ.get("nnUNet_preprocessed")
    results_path = os.environ.get("nnUNet_results")

    assert data_tag in raw_path, f"data_tag {data_tag} not found in nnUNet_raw path {raw_path}"
    assert data_tag in preprocessed_path, f"data_tag {data_tag} not found in nnUNet_preprocessed path {preprocessed_path}"
    assert data_tag in results_path, f"data_tag {data_tag} not found in nnUNet_results path {results_path}"

    for fold in range(5):

        # instantiate the nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, 'Dataset111_CellMapER/nnUNetTrainer__nnUNetPlans__3d_fullres'),
            use_folds=(fold,),
            checkpoint_name='checkpoint_best.pth',
        )

        predictor.predict_from_files(join(nnUNet_raw, 'Dataset111_CellMapER/imagesTr'),
                                        join(nnUNet_results, 'Dataset111_CellMapER', 'nnUNetTrainer__nnUNetPlans__3d_fullres', f'fold_{fold}', 'pred'),
                                        save_probabilities=False, overwrite=False,
                                        num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)