from trident.wsi_objects.CuCIMWSI import CuCIMWSI
from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI

import numpy as np
import cv2

from glob import glob

def visualize_grid(
    thumbnail: np.array, 
    grids, 
    patch_size = 16,
    verbose: int = 0,
    vis_path: str = None,
):
    _vis = np.zeros_like(thumbnail)

    if verbose > 0: print(f'Visualization patch size : {patch_size}')
    
    for x, y in grids:
        _vis[y:y+patch_size, x:x+patch_size] = thumbnail[y:y+patch_size, x:x+patch_size]

    cv2.imwrite(vis_path, _vis)

def get_coords(slide_path: str, patch_size: int = 256, source_mag:int = 20, target_mag:int = 20, artifact_remover_model = None, visualization = None, multi_scale: bool = False, reader = 'cucim', mpp=None):
    # Load CuCim using trident object
    if reader == 'cucim':
        slide = CuCIMWSI(slide_path=slide_path, lazy_init=True, custom_mpp_keys=None, mpp=mpp)
    if reader == 'openslide':
        slide = OpenSlideWSI(slide_path=slide_path, lazy_init=True, custom_mpp_keys=None, mpp=mpp)
        
    # Tissue segmenetation
    gdf_contours = slide.segment_tissue(
        segmentation_model = artifact_remover_model,
        target_mag = target_mag,
        holes_are_tissue = True,
        device = 'cuda',
        job_dir = visualization,
    )

    if visualization is not None:
        gdf_contours = slide.gdf_contours
    
    # count L0 patches
    _dims = slide.level_dimensions
    n_l0_patches = int(_dims[0][0] / patch_size) * int(_dims[0][1] / patch_size)
    
    patcher = slide.create_patcher(
        patch_size = patch_size,
        src_mag = source_mag, # 20
        dst_mag = target_mag,
        mask = gdf_contours,
        coords_only = True,
        overlap = 0,
        threshold = 0,
    )
    
    coords_to_keep = [(x.item(), y.item()) for x, y in patcher]
    
    grids = {
        # Level 0 (20x)
        'l0' : coords_to_keep,
        'n_l0' : n_l0_patches,
    }
    
    if multi_scale:
        # Get L1 level coords
        l1_patcher = slide.create_patcher(
            patch_size = patch_size,
            src_mag = source_mag, # 20
            dst_mag = int(target_mag / 4),
            mask = gdf_contours,
            coords_only = True,
            overlap = 0,
            threshold = 0,
        )
        
        # for L1 image size 
        n_l1_patches = int(_dims[1][0] / patch_size) * int(_dims[1][1] / patch_size)
        l1_coords = [(int(x.item()), int(y.item())) for x, y in l1_patcher] 
        
        grids = {
            # Level 0 (20x)
            'l0' : coords_to_keep,
            'n_l0' : n_l0_patches,
            # Level 1 (5x)
            'l1' : l1_coords,
            'n_l1' : n_l1_patches,
        }

    return grids