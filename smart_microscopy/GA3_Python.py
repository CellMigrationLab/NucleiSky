# IMPORTANT: 'limnode' must be imported like this (not from nor as)
import limnode

# Import other necessary libraries 
from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path
import numpy as np
import tifffile
import shutil
import json
import sys

sys.path.insert(0, r"H:\github\nucleisky-main\src")

# NucleiSky core imports
from nucleisky2d.pipeline import NucleiSky, run_adaptive_matching_and_export
from nucleisky2d.segmentation import segment_nuclei_dispatch
from nucleisky2d.config import DEFAULT_MATCHER_CONFIG
from nucleisky2d.features import extract_nuclear_features,add_centroids_orig_px_columns, extract_centroids_um
from nucleisky2d.preprocess import ij_percentile_normalize, scale_normalize_pair_for_segmentation
        
# Get today's date for output directory naming
today = datetime.now().strftime("%Y%m%d")
output_path = r"G:"
output_dir = Path(output_path) / f"{today}_Experiment" / "local_logging"
output_dir.mkdir(parents=True, exist_ok=True)
        
# NOTE: log from child process
def _log(output_dir, message):
    with open(output_dir / "running.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {message}\n")

# defines output parameter properties
def output(inp: tuple[limnode.AnyInDef], out: tuple[limnode.AnyOutDef]) -> None:    
    pass

# return Program for dimension reduction or two-pass processing
def build(loops: list[limnode.LoopDef]) -> limnode.Program|None:
    return None

# called for each frame/volume
def run(inp: tuple[limnode.AnyInData], out: tuple[limnode.AnyOutData], ctx: limnode.RunContext) -> None:
    
    
    debug = True
    plot = False
    adaptive = False

    _log(output_dir,  "\n" + "="*100 + "\n" + "="*100 + "\n" + f"Run started!")
    
    # Define the input paths and output directory
    input_full_img = inp[0].data[0,:,:,0]
    input_full_pixel_size = inp[0].calibration[0]
    input_crop_img = inp[1].data[0,:,:,0]
    input_crop_pixel_size = inp[1].calibration[0]
    
    if debug:
        tifffile.imwrite(output_dir / "input_full_img.tif", input_full_img)
        tifffile.imwrite(output_dir / "input_crop_img.tif", input_crop_img)

        _log(output_dir, f"Chosen configuration:")
        _log(output_dir, f"  Full image: {input_full_img.shape} with pixel size {input_full_pixel_size} µm")
        _log(output_dir, f"  Crop image: {input_crop_img.shape} with pixel size {input_crop_pixel_size} µm")
        _log(output_dir, f"  Output directory: {output_dir}")
        
        if plot:
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(input_full_img)
            plt.title('Full input image')
            plt.title('Full image')
            plt.subplot(1,2,2)
            plt.imshow(input_crop_img)
            plt.title('Crop image')
            plt.show()
        
    _log(output_dir, "Normalizing scales for segmentation...")
    (img_full_seg, img_crop_seg,
    ps_full_seg, ps_crop_seg,
    scale_f, scale_c, target_um) = scale_normalize_pair_for_segmentation(
        input_full_img, input_crop_img,
        input_full_pixel_size, input_crop_pixel_size
    )
    _log(output_dir, f"Normalizing done!")

    # Define segmentation settings
    seg_method = "threshold"  # Options: "threshold", "cellpose", "instanseg"
    seg_settings = {
        "threshold": {
            "threshold_method": "otsu",
            "min_object_size": 5,
            "do_watershed": True,
        }
    }

    _log(output_dir, f"Segmenting using {seg_method}...")
    masks_full = segment_nuclei_dispatch(img_full_seg, seg_method, ps_full_seg, seg_settings)
    masks_crop = segment_nuclei_dispatch(img_crop_seg, seg_method, ps_crop_seg, seg_settings)
    _log(output_dir, "Segmentation done!")

    if debug:
        _log(output_dir, f"Full mask shape: {masks_full.shape}")
        _log(output_dir, f"Full mask mean: {masks_full.mean()}")
        _log(output_dir, f"Crop mask shape: {masks_crop.shape}")
        _log(output_dir, f"Crop mask mean: {masks_crop.mean()}")
        
        tifffile.imwrite(output_dir / "full_mask.tif", masks_full)
        tifffile.imwrite(output_dir / "crop_mask.tif", masks_crop)
        
        if plot:
            plt.figure(figsize=(12,12))
            plt.subplot(2,2,1)
            plt.imshow(input_full_img)
            plt.title('Full input image')
            plt.subplot(2,2,2)
            plt.imshow(input_crop_img)
            plt.title('Crop image')    
            plt.subplot(2,2,3)
            plt.imshow(masks_full)
            plt.title('Full input mask')
            plt.subplot(2,2,4)
            plt.imshow(masks_crop)
            plt.title('Crop mask')
            plt.show()
        
    _log(output_dir, "Extracting features...")
    df_full = extract_nuclear_features(masks_full, None, ps_full_seg)
    df_crop = extract_nuclear_features(masks_crop, None, ps_crop_seg)
    _log(output_dir, "Feature extraction done!")

    # Map coordinates back to original pixel space
    _log(output_dir, "Calculating centroids in original pixel space...")
    df_full = add_centroids_orig_px_columns(df_full, scale_f)
    df_crop = add_centroids_orig_px_columns(df_crop, scale_c)
    df_full.to_csv(output_dir / "full_df.csv", index=False)
    df_crop.to_csv(output_dir / "crop_df.csv", index=False)
    _log(output_dir, "Centroids calculation done!")

    if adaptive:
        _log(output_dir, "Running Adaptive Pipeline...")
        with open(output_dir / "running.log", "a", encoding="utf-8") as sys.stdout:
            best_result_adaptive, history = run_adaptive_matching_and_export(
                df_full=df_full,
                df_crop=df_crop,
                img_full=input_full_img,
                img_crop=input_crop_img,
                pixel_size_full_um=input_full_pixel_size,
                pixel_size_crop_um=input_crop_pixel_size,
                result_dir=str(output_dir),
                store_full_out=True,
            )
        _log(output_dir, "Adaptive Pipeline done!")

        if best_result_adaptive['success']:
            _log(output_dir, "\nAdaptive Match Success!")
            _log(output_dir, f"Winning Matcher: {best_result_adaptive['matcher']}")
        else:
            _log(output_dir, "\nAdaptive Match Failed.")

    else:
        # 1. Customize configuration
        custom_config = DEFAULT_MATCHER_CONFIG.copy()
        custom_config['quad'] = custom_config.get('quad', {}).copy()

        # Override: Restrict max rotation to +/- 20 degrees (default is 180)
        custom_config['quad']['angle_max_deg'] = 20.0

        # Override: Require at least 30% inliers for a successful match (default is 0.6 or 60%)
        custom_config['_common']['frac_inliers_thresh'] = 0.3  

        # 2. Prepare data (extract µm coordinates)
        centroids_f_um = extract_centroids_um(df_full, name="df_full")
        centroids_c_um = extract_centroids_um(df_crop, name="df_crop")

        _log(output_dir, "Running Manual Triangle Matcher...")
        with open(output_dir / "running.log", "a", encoding="utf-8") as sys.stdout:
            triangle_result = NucleiSky(
                centroids_crop_um=centroids_c_um,
                centroids_full_um=centroids_f_um,
                img_full=input_full_img,
                img_crop=input_crop_img,
                ij_percentile_normalize=ij_percentile_normalize,
                pixel_size_full_um=input_full_pixel_size,
                pixel_size_crop_um=input_crop_pixel_size,
                matcher="triangles",
                matcher_config=custom_config,
                df_full=df_full,
                df_crop=df_crop,
            )
        _log(output_dir, "Triangle Matcher done!")

        if triangle_result['success']:
            _log(output_dir, "\nTriangle Match Success!")
            best_result_adaptive = triangle_result
        else:
            _log(output_dir, "\nTriangle Match Failed. Trying Quad Matcher with same angle restriction...")
            _log(output_dir, "Running Manual QUAD Matcher (Max Angle = 20°)...")
            with open(output_dir / "running.log", "a", encoding="utf-8") as sys.stdout:
                best_result_adaptive = NucleiSky(
                    centroids_crop_um=centroids_c_um,
                    centroids_full_um=centroids_f_um,
                    img_full=input_full_img,
                    img_crop=input_crop_img,
                    ij_percentile_normalize=ij_percentile_normalize,
                    pixel_size_full_um=input_full_pixel_size,
                    pixel_size_crop_um=input_crop_pixel_size,
                    matcher="quad",
                    matcher_config=custom_config,
                    df_full=df_full,
                    df_crop=df_crop,
                )
            _log(output_dir, "Quad Matcher done!")
            
            if best_result_adaptive['success']:
                _log(output_dir, "\nQuad Match Success!")
            else:
                _log(output_dir, "\nQuad Match Failed.")
    
    ## Define the outputs of the pipeline
    # Get the success flagmatcher
    if best_result_adaptive['success']:
        success = 1 
        
        # Get the best rotation matrix and flatten it to a list for output
        best_rotation = best_result_adaptive['best_R']
        best_rotation_flat = best_rotation.flatten().tolist()
        if any([np.abs(val) > 0.1 for val in best_rotation_flat]):
            rotated = 1
        else:
            rotated = 0
        
        # Get the centroid from the bounding box of the matched crop in the full image (if available)
        bbox = best_result_adaptive['bbox_full_px']
        bbox_centroid_x = bbox.x0 + (bbox.x1 - bbox.x0) / 2
        bbox_centroid_y = bbox.y0 + (bbox.y1 - bbox.y0) / 2
    else:
        success = 0
        rotated = 0
        bbox_centroid_x = 0.0
        bbox_centroid_y = 0.0

    _log(output_dir, f"Output values are: Success={success}, Rotated={rotated}, BBox Centroid=({bbox_centroid_x}, {bbox_centroid_y})")

    # Save the outputs to the expected format
    output_dict = {
        "Success_flag":int(success), 
        "Rotated_flag":int(rotated), 
        "X_coord":float(bbox_centroid_x), 
        "Y_coord":float(bbox_centroid_y)
    }
    
    with open(output_dir / "tmp.yaml", "w", encoding="utf-8") as file:
        json.dump(output_dict, file, indent=2)

# child process initialization (when outproc is set)
if __name__ == '__main__':
    limnode.child_main(run, output, build)