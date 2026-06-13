# torch
import torch
from torch.utils.data import Dataset

# Common
import os
import pandas as pd
from tqdm import tqdm

# Multi-processing
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List 
from laguadia.utils.filtering import get_coords
import cv2
import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image


__all__ = ['LaGuadiaStage1Dataset', 'LaGuadiaStage2Dataset', 'PatchExtractionDataSet']

class LaGuadiaStage1Dataset(Dataset):
    def __init__(
        self,
        args,
        dfs: pd.DataFrame,
        keyword_bank: dict,
        tokenizer,
        fold: int,
        split: str,
    ):
        super().__init__()
        self.args = args
        self.root_dir = args.root_dir
        self.data_type = args.data_type
        self.verbose = args.verbose
        self.split = split
        self.fold = fold
        
        # Split Data Frame
        if args.data_type.lower() == 'tcga':
            self.df = dfs[dfs[f'fold_{self.fold}'] == self.split]
        else:
            self.df = dfs[dfs[f'split'] == self.split]

        # Load Features
        self.svs_list = self.df['p_id'].tolist() # [:2]
        
        files_list = os.listdir(os.path.join(self.root_dir, 'gigapath'))
        file_df = pd.DataFrame({'names' : files_list})

        # Meta-teacher Processor
        self.tokenizer = tokenizer

        # Keyword Bank
        self.keyword_bank = keyword_bank

        # Keywords
        self.keywords = {}
        self.nt_ids = {}
        
        for _item in self.df.iloc:
            texts = []
            for _t in _item.keywords.split(','):
                texts.append(_t.lower().strip())
            texts.append("normal tissue")

            _text = None
            for _t in texts:
                _k = keyword_bank.get(_t, None)
                
                if _k == None:
                    print(f'{_t} is Not in Keyword Bank')
                    continue
                
                _text = _k if _text is None else torch.vstack([_text, _k])
            
            MAX_PADDING_CNT = 15
            
            self.nt_ids[_item.p_id] = _text.shape[0]
            
            if _text.shape[0] < MAX_PADDING_CNT:
                _padding_cnt = MAX_PADDING_CNT - _text.shape[0]
                _text = torch.vstack([_text, torch.ones((_padding_cnt, _text.shape[1]), dtype=torch.int)])

            self.keywords[_item.p_id] = _text
        
        self.teachers = []
        self.features = {}

        if args.use_gigapath: self.teachers.append('gigapath')
        if args.use_uni: self.teachers.append('uni')
        if args.use_virchow2: self.teachers.append('virchow2')
        
        self.teachers.append('medgemma')
        empty_list = []
        
        print(f'Teachers {self.teachers}')
        
        for i, teacher in enumerate(self.teachers):
            self.features[teacher] = {}
                    
            # For multi threading
            def load_features(svs_name: str):
                _inner_file_list = file_df[file_df['names'].str.contains(svs_name)]['names'].tolist()
                
                if len(_inner_file_list) == 0:
                    self.print(f'File not found : {svs_name}')
                    return svs_name
                
                _new_dict = {}
                for i, _feature_file_name in enumerate(_inner_file_list):
                    _dir = os.path.join(self.root_dir, teacher, _feature_file_name)
                    _feature_data = torch.load(_dir)
                    
                    for _key in _feature_data.keys():
                        _new_key = _key.replace('0_', f'{i}_')
                        _new_dict[f'{svs_name}_{_new_key}'] = _feature_data[_key]
                        
                return _new_dict
            
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(load_features, svs_name) for svs_name in self.svs_list]
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"[{i + 1} / {len(self.teachers)}] Reading {teacher} features"):
                    _feature_dict = f.result()
                    
                    if type(_feature_dict) is str:
                        print(f'[Error] {_feature_dict} is empty')
                        empty_list.append(_feature_dict)
                        continue
                    
                    self.features[teacher].update(_feature_dict)

        self.feature_keys = list(self.features['medgemma'].keys())
            
        print(f'Skipped {len(empty_list)} samples')
            
    def __getitem__(self, idx):
        _key = self.feature_keys[idx]
        _svs_name = _key.split('_')[0]
        outputs = {'key' : _key, 'svs_name' : _svs_name}
        
        for teacher in self.teachers: # ['UNI', 'GigaPath', 'Virchow2', 'MedGemma]
            outputs[f'{teacher}_feats'] = self.features[teacher][_key]
        
        outputs['texts'] = self.keywords.get(_svs_name, None)
        outputs['nt_idxs'] = self.nt_ids.get(_svs_name, None)
        return outputs

    def __len__(self):
        return len(self.feature_keys)

    def print(self, text: str, level: int = 1):
        if self.verbose >= level:
            print(text)

import torch.nn as nn
from glob import glob

from cucim import CuImage
import cupy as cp
import numpy as np

class LaGuadiaStage2Dataset(Dataset):
    def __init__(
        self,
        args,
        dfs: pd.DataFrame,
        keyword_bank: dict,
        tokenizer,
        fold: int,
        split: str,
        dinov3_processor,
        artifact_remover_model: nn.Module = None, 
    ):
        super().__init__()
        self.args = args
        self.root_dir = args.root_dir
        self.data_type = args.data_type
        self.verbose = args.verbose
        self.patch_size = args.patch_size
        self.split = split
        self.fold = fold
        self.dinov3_processor = dinov3_processor    
        
        self.cache_svs = args.get('cache_svs', False)    
        
        # CLAM
        self.artifact_remover_model = artifact_remover_model
        
        # Split Data Frame
        if args.data_type.lower() == 'tcga':
            self.df = dfs[dfs[f'fold_{self.fold}'] == self.split]
        else:
            self.df = dfs[dfs[f'split'] == self.split]

        # Load Features # 
        self.svs_list = sorted(self.df['p_id'].tolist()) # [:2]
        
        files_list = os.listdir(os.path.join(self.root_dir, 'gigapath'))
        file_df = pd.DataFrame({'names' : files_list})

        # Meta-teacher Processor
        self.tokenizer = tokenizer

        # CuCim Object 대신 patch load 방식 구현
        self.cucim_objects = {}
        
        skip_pid = []
        for p_id in tqdm(self.svs_list, desc = '[Loading CuCIM Objects]'):
            target_files = glob(os.path.join(args.svs_dir, f'{p_id}*.svs'))
            target_files = sorted(target_files)
            for i, svs_file_name in enumerate(target_files):
                try:
                    self.cucim_objects[f'{p_id}_{i}'] = CuImage(svs_file_name) # 'p_id_i'
                except Exception as e:
                    print(f'Error while loading {svs_file_name} : {e}')
                    skip_pid.append(p_id)
        
        # Keyword Bank
        self.keyword_bank = keyword_bank

        # Keywords
        self.keywords = {}
        self.nt_ids = {}
        
        for _item in self.df.iloc:
            texts = []
            for _t in _item.keywords.split(','):
                texts.append(_t.lower().strip())
            texts.append("normal tissue")

            _text = None
            for _t in texts:
                _k = keyword_bank.get(_t, None)
                
                if _k == None:
                    print(f'{_t} is Not in Keyword Bank')
                    continue
                
                _text = _k if _text is None else torch.vstack([_text, _k])
            
            MAX_PADDING_CNT = 15
            
            self.nt_ids[_item.p_id] = _text.shape[0]
            
            if _text.shape[0] < MAX_PADDING_CNT:
                _padding_cnt = MAX_PADDING_CNT - _text.shape[0]
                _text = torch.vstack([_text, torch.ones((_padding_cnt, _text.shape[1]), dtype=torch.int)])

            self.keywords[_item.p_id] = _text
        
        self.teachers = []
        self.features = {}

        if args.use_gigapath: self.teachers.append('gigapath')
        if args.use_uni: self.teachers.append('uni')
        if args.use_virchow2: self.teachers.append('virchow2')
        
        self.teachers.append('medgemma')
        empty_list = []
        
        print(f'Teachers {self.teachers}')
        
        self.patch_regions = {}
        
        for i, teacher in enumerate(self.teachers):
            self.features[teacher] = {}

            # For multi threading
            def load_features(svs_name: str):
                if svs_name in skip_pid:
                    self.print(f'Skip due to loading error from SVS file : {svs_name}')
                    return svs_name
                
                _inner_file_list = file_df[file_df['names'].str.contains(svs_name)]['names'].tolist()
                _inner_file_list = sorted(_inner_file_list)
                
                if len(_inner_file_list) == 0:
                    self.print(f'File not found : {svs_name}')
                    skip_pid.append(svs_name)
                    return svs_name
                
                # _data = []
                _new_dict = {}
                for i, _feature_file_name in enumerate(_inner_file_list):
                    _dir = os.path.join(self.root_dir, teacher, _feature_file_name)
                    _feature_data = torch.load(_dir)
                    
                    _real_svs_name = _feature_file_name.replace('.pt', '')
                    
                    self.patch_regions[f'{svs_name}_{i}'] = [] # ((x,y), 'Mapping Name')
                    
                    for _key in _feature_data.keys():
                        _coords = _key.split('_')[1]
                        _x, _y = int(_coords.split('x')[0]), int(_coords.split('x')[1])
                        
                        _new_key = _key.replace('0_', f'{i}_')
                        _new_dict[f'{svs_name}_{_new_key}'] = _feature_data[_key]
                        
                        self.patch_regions[f'{svs_name}_{i}'].append(((_x,_y), f'{svs_name}_{_new_key}'))
                        
                return _new_dict
            
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(load_features, svs_name) for svs_name in self.svs_list]
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"[{i + 1} / {len(self.teachers)}] Reading {teacher} features"):
                    _feature_dict = f.result()
                    
                    if type(_feature_dict) is str:
                        # print(f'Error while {_feature_dict} is Empty')
                        empty_list.append(_feature_dict)
                        continue
                    
                    self.features[teacher].update(_feature_dict)

        self.feature_keys = list(self.features['medgemma'].keys())
        
        # Loading Raw Patches (TCGA Only)
        if self.cache_svs:
            self.patches = {}
            
            _cucim_keys = self.cucim_objects.keys()
            
            self.patch_regions_keys = list(self.patch_regions.keys())
            
            for p_id in tqdm(self.svs_list, desc = '[Loading Patches]'):
                if p_id in skip_pid:
                    print(f'[Skip] {p_id}')
                    continue

                seledted_keys = [s for s in _cucim_keys if p_id in s]                        
                
                for i, svs_key_name in enumerate(seledted_keys):
                    self.print(f'Load WSI [{i+1} / {len(seledted_keys)}] Load from : {svs_key_name}')
                    
                    if svs_key_name not in self.patch_regions_keys:
                        print(f'[Skip] {svs_key_name} not in patch_region_keys')
                        continue
                    
                    # TCGA
                    slide = self.cucim_objects[svs_key_name]
                    _svs_coords_info = self.patch_regions[svs_key_name]
                    
                    def read_l0_patch(_coord_info_cahce):
                        x, y = _coord_info_cahce[0]
                        _key = _coord_info_cahce[1]
                        patch = slide.read_region(location=(x, y), size=(self.patch_size, self.patch_size), level=0)
                        return _key, self.dinov3_processor(cp.asnumpy(patch).astype("uint8"))['pixel_values'][0]
                
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        futures = [executor.submit(read_l0_patch, coord) for coord in _svs_coords_info]
                        for f in tqdm(as_completed(futures), total=len(futures), desc="Reading L0 patches"):
                            key, patch = f.result()
                            self.patches[key] = patch # TCGA-BJ-A0Z0_0_29568x66304
                
        print(f'Skipped {len(empty_list)} samples')
            
    def __getitem__(self, idx):
        _key = self.feature_keys[idx]
        _svs_name = _key.split('_')[0]
        _index = _key.split('_')[1]
        _x, _y = _key.split('_')[2].split('x')
        
        outputs = {'key' : _key, 'svs_name' : _svs_name}
        
        for teacher in self.teachers: # ['UNI', 'GigaPath', 'MedGemma]
            outputs[f'{teacher}_feats'] = self.features[teacher][_key]
        
        if not self.cache_svs:
            _raw_patch_cp = self.cucim_objects[f'{_svs_name}_{_index}'].read_region(location=(int(_x), int(_y)), size=(self.patch_size, self.patch_size), level=0)
            _raw_patch = cp.asnumpy(_raw_patch_cp)
            if _raw_patch.dtype != np.uint8:
                _raw_patch = _raw_patch.astype(np.uint8, copy=False)
            outputs['dinov3_images'] = self.dinov3_processor(_raw_patch)['pixel_values'][0]
            del _raw_patch_cp
            del _raw_patch
        else:
            outputs['dinov3_images'] = self.patches[_key]
        
        outputs['texts'] = self.keywords.get(_svs_name, None)
        outputs['nt_idxs'] = self.nt_ids.get(_svs_name, None)
        return outputs

    def __len__(self):
        return len(self.feature_keys)

    def print(self, text: str, level: int = 1):
        if self.verbose >= level:
            print(text)

class PatchExtractionDataSet(Dataset):
    def __init__(
        self, 
        args, 
        target_levels: List[str] = ['l0', 'l1'],
        svs_names = None,
        artifact_remover_model = None,
        save_raw_patch = False,
        pre_coords = None,
        
        # Processors
        medgemma_processor = None,
        dinov3_processor = None,
        virchow2_processor = None, 
        h0mini_processor = None, 
        **kwargs,
    ):
        self.args = args
        self.root_dir = args.root_dir
        self.patch_size = args.patch_size
        self.verbose = args.verbose
        self.data_type = args.data_type.upper()
        self.target_levels = target_levels
        self.save_raw_patch = save_raw_patch
        self.medgemma_processor = medgemma_processor 
        self.dinov3_processor = dinov3_processor
        
        # Check valid file path
        self.svs_names = None
        
        self.root_dir_files = glob(os.path.join(args.root_dir + '/*.svs'))

        if svs_names is not None:
            self.svs_names = svs_names if not type(svs_names) ==  str else [svs_names]
        else:
            self.svs_names = args.svs_names

        if type(self.svs_names) == str:
            self.svs_names = self.svs_names.split(',')
        
        valid_svs_name = []
        for svs_name in self.svs_names:
            for _file in self.root_dir_files:
                if svs_name in _file:
                    valid_svs_name.append(_file)
                    continue
                
        self.svs_names = valid_svs_name

        self.patches = {}
        for target_l in self.target_levels:
            self.patches[target_l] = {}
        
        # Processing Slide Object
        for i, svs_name in enumerate(self.svs_names):
            _name = os.path.basename(svs_name).split('.')[0]
            if f'{_name}' in os.listdir(self.args.root_extract_dir):
                print(f'Skip {_name}')
                continue
            
            self.print(f'Load WSI [{i+1} / {len(self.svs_names)}] Load from : {svs_name}')
            
            # Open slide object with CuCim
            try:
                slide = CuImage(os.path.join(self.root_dir, svs_name))
                path_to_slide = os.path.join(self.root_dir, svs_name)
            except ValueError as e:
                print(f'Skip {svs_name} due to error : {e}')
                continue
                
            if self.verbose > 0:
                print(slide.resolutions["level_dimensions"])
                print(slide.resolutions["level_downsamples"])
            
            # Get CLAM based coords
            if pre_coords is None:
                try:
                    ms_grids = get_coords(
                        slide_path = path_to_slide,
                        source_mag=20, # TCGA : 40
                        target_mag=20,
                        patch_size = self.patch_size,
                        artifact_remover_model = artifact_remover_model,
                        visualization = os.path.join(args.root_extract_dir, 'vis'),
                        multi_scale = False, # Get L0 and L1 coords
                    )
                except Exception as e:
                    ms_grids = get_coords(
                        slide_path = path_to_slide,
                        source_mag=20, # TCGA : 40
                        target_mag=20,
                        patch_size = self.patch_size,
                        artifact_remover_model = artifact_remover_model,
                        visualization = os.path.join(args.root_extract_dir, 'vis'),
                        multi_scale = False, # Get L0 and L1 coords
                        mpp = 0.5,
                    )
                    print(f'Using MPP 0.5 for {svs_name} due to error: {e}')
            else:
                ms_grids = {'l0' : pre_coords, 'n_l0': len(pre_coords)}
                
            if self.save_raw_patch:
                _name = os.path.basename(svs_name).split('.')[0]
                os.makedirs(os.path.join(args.root_extract_dir, _name), exist_ok=True)
                os.makedirs(os.path.join(args.root_extract_dir, _name, 'l0'), exist_ok=True)
                os.makedirs(os.path.join(args.root_extract_dir, _name, 'l1'), exist_ok=True)
                os.makedirs(os.path.join(args.root_extract_dir, _name, 'l2'), exist_ok=True)
                
            def read_patch(coord, level):
                x, y = coord
                patch = slide.read_region(location=(x, y), size=(self.patch_size, self.patch_size), level=level)
                
                if self.save_raw_patch:
                    cv2.imwrite(os.path.join(args.root_extract_dir, _name) + f'/l{level}/{x}_{y}.png', cp.asnumpy(patch).astype("uint8"))
                
                return f'{i}_{x}x{y}', cp.asnumpy(patch).astype("uint8")

            if 'l0' in self.target_levels:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(read_patch, coord, 0) for coord in ms_grids['l0']]
                    for f in tqdm(as_completed(futures), total=len(futures), desc="Reading L0 patches"):
                        key, patch = f.result()
                        self.patches['l0'][key] = patch
            
            if 'l1' in self.target_levels:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(read_patch, coord, 1) for coord in ms_grids['l1']]
                    for f in tqdm(as_completed(futures), total=len(futures), desc="Reading L1 patches"):
                        key, patch = f.result()
                        self.patches['l1'][key] = patch
                        
            if 'l2' in self.target_levels:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(read_patch, coord, 2) for coord in ms_grids['l2']]
                    for f in tqdm(as_completed(futures), total=len(futures), desc="Reading L2 patches"):
                        key, patch = f.result()
                        self.patches['l2'][key] = patch

            if 'l0' in self.target_levels: self.print(f'L0 Full grid count : {ms_grids["n_l0"]}, Filtered gird count : {len(self.patches["l0"].keys())}')
            if 'l1' in self.target_levels: self.print(f'L1 Full grid count : {ms_grids["n_l1"]}, Filtered gird count : {len(self.patches["l1"].keys())}')
            if 'l2' in self.target_levels: self.print(f'L2 Full grid count : {ms_grids["n_l2"]}, Filtered gird count : {len(self.patches["l2"].keys())}')

        self.l0_list = list(self.patches['l0'].keys())
        
        # Define Processors
        self.processors = {}
        if not self.save_raw_patch:
            # ===== GigaPath transform processor ===== 
            if args.use_gigapath:
                self.processors['gigapath'] = transforms.Compose(
                    [
                        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
            
            # ===== UNI transform processor =====
            if args.use_uni:
                self.processors['uni'] = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )

            # ===== MUSK transform processor =====
            if args.use_musk:
                self.processors['musk'] = transforms.Compose(
                    [
                        transforms.Resize(384, interpolation=3, antialias=True),
                        transforms.CenterCrop((384, 384)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
                    ]
                )
                
            if args.get('use_virchow2', False):
                self.processors['virchow2'] = virchow2_processor

            if args.get('use_h0mini', False):
                self.processors['h0mini'] = h0mini_processor
            
            if args.get('use_gpfm', False):
                self.processors['gpfm'] = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                            
            if args.use_medgemma:
                self.medgemma_resize = transforms.Resize((448, 448),interpolation=transforms.InterpolationMode.BILINEAR)
                
    def __getitem__(self, idx):
        _key = self.l0_list[idx]
        patch = self.patches['l0'][_key]
        torch_patch = torch.from_numpy(patch)
        pil_image = Image.fromarray(patch)
        outputs = {}
        outputs['key'] = _key
        
        if self.dinov3_processor is not None: 
            outputs['dinov3_images'] = self.dinov3_processor(images=torch_patch, return_tensors="pt")['pixel_values'][0]
        
        if self.medgemma_processor is not None:
            outputs['medgemma_images'] = self.medgemma_processor(patch, return_tensors='pt')['pixel_values'].squeeze(0)
        
        for encoder_model_key in self.processors.keys():
            outputs[f'{encoder_model_key}_images'] = self.processors[encoder_model_key](pil_image)
        
        return outputs
        
    def __len__(self):
        return len(self.l0_list)
    
    def print(self, text: str, level: int = 1):
        if self.verbose >= level:
            print(text)