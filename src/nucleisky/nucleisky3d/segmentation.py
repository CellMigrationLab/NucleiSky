"""segmentation.py 3D segmentation backends."""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix

if find_spec("tqdm") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    _tqdm = None

from nucleisky2d.segmentation import get_global_segmentor, segment_nuclei_dispatch

logger = logging.getLogger(__name__)


def stitch_2d_slices(
    slice_labels: list[np.ndarray],
    min_iou: float = 0.3,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Stitches a list of 2D label arrays into a 3D volume using IoU overlap.
    Highly optimized using Scipy Sparse matrices to compute intersections in O(N).
    """
    if not slice_labels:
        return np.array([], dtype=np.int32)
        
    shape = (len(slice_labels), *slice_labels[0].shape)
    vol_3d = np.zeros(shape, dtype=np.int32)
    
    # Base case: first slice
    vol_3d[0] = slice_labels[0]
    current_max_id = int(vol_3d[0].max())
    
    z_slices = range(1, len(slice_labels))
    if show_progress and _tqdm is not None:
        z_slices = _tqdm(z_slices, desc="Stitching slices into 3D")

    for z in z_slices:
        prev_lbl = vol_3d[z-1]
        curr_lbl = slice_labels[z]
        
        # Flatten arrays for fast 1D operations
        p_flat = prev_lbl.ravel()
        c_flat = curr_lbl.ravel()
        
        # Mask out pixels that are background in both slices to speed up processing
        fg_mask = (p_flat > 0) | (c_flat > 0)
        if not np.any(fg_mask):
            continue  # Both slices are entirely empty
            
        p_fg = p_flat[fg_mask]
        c_fg = c_flat[fg_mask]
        
        max_p = p_fg.max()
        max_c = c_fg.max()
        
        # Array to hold the new IDs. Index = old ID, Value = new ID
        mapping = np.zeros(max_c + 1, dtype=np.int32)
        
        # If previous slice was empty, all current cells get new IDs
        if max_p == 0:
            curr_ids = np.unique(c_fg[c_fg > 0])
            for cid in curr_ids:
                current_max_id += 1
                mapping[cid] = current_max_id
            vol_3d[z] = mapping[curr_lbl]
            continue
            
        # 1. Compute areas of all cells in one pass
        area_prev = np.bincount(p_flat, minlength=max_p + 1)
        area_curr = np.bincount(c_flat, minlength=max_c + 1)
        
        # 2. Compute intersection matrix of ALL cells simultaneously
        intersections_csc = coo_matrix(
            (np.ones(len(p_fg), dtype=np.int32), (p_fg, c_fg)), 
            shape=(max_p + 1, max_c + 1)
        ).tocsc()
        
        # 3. Determine new IDs
        curr_ids = np.unique(c_fg)
        curr_ids = curr_ids[curr_ids > 0] # Skip background
        
        for cid in curr_ids:
            col = intersections_csc[:, cid]
            prev_ids_overlap = col.indices
            overlaps = col.data
            
            valid_mask = prev_ids_overlap > 0
            prev_ids_overlap = prev_ids_overlap[valid_mask]
            overlaps = overlaps[valid_mask]
            
            if len(prev_ids_overlap) > 0:
                best_idx = np.argmax(overlaps)
                best_prev_id = prev_ids_overlap[best_idx]
                best_intersection = overlaps[best_idx]
                
                union = area_prev[best_prev_id] + area_curr[cid] - best_intersection
                iou = best_intersection / union
                
                if iou >= min_iou:
                    mapping[cid] = best_prev_id
                else:
                    current_max_id += 1
                    mapping[cid] = current_max_id
            else:
                current_max_id += 1
                mapping[cid] = current_max_id
                
        # 4. Apply mapping to the entire current slice instantly
        vol_3d[z] = mapping[curr_lbl]
        
    return vol_3d


def segment_nuclei_2p5d(
    volume_zyx: np.ndarray,
    method: str,
    pixel_size_um_zyx: tuple[float, float, float],
    settings: dict = None,
    min_iou: float = 0.3,
    show_progress: bool = True,
    segmentor: Optional[object] = None,
) -> np.ndarray:
    """
    Executes 2.5D segmentation: runs a 2D segmentation method on every Z-slice,
    then stitches them together into a 3D volume.
    """
    settings = settings or {}
    volume_zyx = np.asarray(volume_zyx)

    # Use the mean of Y and X dimensions for a more robust 2D pixel size
    pixel_size_2d = float(np.mean(pixel_size_um_zyx[1:]))

    seg_impl = segmentor if segmentor is not None else get_global_segmentor()

    slice_labels = []
    z_slices = range(volume_zyx.shape[0])
    if show_progress and _tqdm is not None:
        z_slices = _tqdm(z_slices, desc=f"Segmenting 2D slices ({method})")

    # 1. Segment slice by slice using your existing nucleisky2d dispatch
    for z in z_slices:
        slice_img = volume_zyx[z]
        lbl_2d = segment_nuclei_dispatch(
            img=slice_img,
            method=method,
            pixel_size_um=pixel_size_2d,
            settings=settings,
            segmentor=seg_impl,
        )
        slice_labels.append(lbl_2d)

    # 2. Stitch the 2D slices together
    logger.info(f"Stitching 2D slices into 3D (min_iou={min_iou})...")
    labels_3d = stitch_2d_slices(
        slice_labels,
        min_iou=min_iou,
        show_progress=show_progress,
    )

    return labels_3d
