import os
import time
import hydra
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import AutoImageProcessor

# CLAM
from trident.segmentation_models import segmentation_model_factory

# Lab Model
from laguadia import LaGuadiaModel
from laguadia.datasets import PatchExtractionDataSet
from laguadia.utils import seed_everything

if hasattr(torch, "compiler") and not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

seed_everything(1813)

@hydra.main(config_path = "./configs", version_base = None)
def main(configs):
    # Load Pretrained LaGuadia
    print(f'Loading model from {configs.pretrained}')
    state_dict = torch.load(configs.pretrained, weights_only=False)
    
    # Check Path
    if not os.path.exists(configs.root_extract_dir):
        os.makedirs(configs.root_extract_dir, exist_ok=True)
    
    wsi_names = os.listdir(configs.root_dir)
    wsi_names = [_f.replace('.svs', '') for _f in wsi_names]
        
    print(f'=========== Processing information ===========')
    print(f'Target csv : {len(wsi_names)}')

    # Load segment model
    artifact_remover_model = segmentation_model_factory('grandqc_artifact')
    artifact_remover_model.cuda()
    
    # Load DINOv3 Processor
    dinov3_processor = AutoImageProcessor.from_pretrained(configs.pretrained_model_name)

    # Init LaGuadia Model
    model = LaGuadiaModel(configs)
    model.cuda()
    model.eval()
    
    model.load_state_dict(state_dict['state_dict'], strict=False)
    
    for slide_name in tqdm(wsi_names, desc=f'[extraction]'):

        # Get Coords By Name
        ext_keys = None
        if configs.ref_file_dir is not None:
            f_name = slide_name.split('/')[-1]
            _ref_file_path = os.path.join(configs.ref_file_dir, f'{f_name}.pt')
            print(f'Find refer file from : {_ref_file_path}')
            if os.path.exists(_ref_file_path):
                print(f'Found reference file for {slide_name}')
                ref_data = torch.load(_ref_file_path)
                ext_keys = []

                for _key in ref_data.keys():
                    ext_keys.append((int(_key.split('_')[1].split('x')[0]), int(_key.split('_')[1].split('x')[1])))
                                
        
        # Patient-Level Dataset
        wsi_dataset = PatchExtractionDataSet(configs, target_levels = ['l0'], svs_names = slide_name, artifact_remover_model = artifact_remover_model, dinov3_processor = dinov3_processor, pre_coords = ext_keys)
        wsi_loader = DataLoader(wsi_dataset, batch_size=int(configs.encoder_batch), num_workers=8, shuffle = False)
        
        wsi_bar = tqdm(wsi_loader)
        
        dwkd_dict = {}
        
        with torch.inference_mode():
            for i, inputs in enumerate(wsi_bar):
                features = model(inputs['dinov3_images'].cuda(non_blocking=True)).cpu()

                for i, _key in enumerate(inputs['key']):
                    dwkd_dict[_key] = features[i]

        torch.save(dwkd_dict, configs.root_extract_dir + f'/f{slide_name}.pt') 

if __name__ == "__main__":
    main()