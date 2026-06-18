# IMPORTANT: 'limjob' must be imported like this (not from nor as)
import limjob

from datetime import datetime
from pathlib import Path
import time
import json

def run(imgs: tuple[limjob.Image], Job: limjob.JobParam, macro: limjob.MacroParam, ctx: limjob.RunContext):
    today = datetime.now().strftime("%Y%m%d")
    output_path = r"G:"
    output_dir = Path(output_path) / f"{today}_Experiment" / "local_logging"

    tmp_file_path = output_dir / "tmp.yaml"
    
    cont = 0
    while not tmp_file_path.exists() and cont < 18:
        time.sleep(10)
        cont += 1 
    
    if tmp_file_path.exists():
        with open(tmp_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        Job.PythonScript.Success_flag = data["Success_flag"]
        Job.PythonScript.Rotated_flag = data["Rotated_flag"]
        x_coord_pixels = data["X_coord"]
        y_coord_pixels = data["Y_coord"]
        # image_width_pixels = data["Image_width"]
        # image_height_pixels = data["Image_height"]
        # image_center_x_pixels = image_width_pixels // 2 
        # image_center_y_pixels = image_height_pixels // 2 
    
        # stage_x_micrometer = Job.Points.CurrentPoint.Position.Stage.x    
        # stage_y_micrometer = Job.Points.CurrentPoint.Position.Stage.y
        
        img = imgs[0]
        # cal_x, cal_y, cal_z = img.calibration
        
        x_stage_um, y_stage_um = img.transformPxToStage(x_coord_pixels, y_coord_pixels)
        
        Job.PythonScript.X_coord = x_stage_um
        Job.PythonScript.Y_coord = y_stage_um
        
        data["x_stage_um"] = x_stage_um
        data["y_stage_um"] = y_stage_um
        
        with open(tmp_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        Job.PythonScript.Success_flag = 0
        Job.PythonScript.Rotated_flag = 0
        Job.PythonScript.X_coord = 0.0
        Job.PythonScript.Y_coord = 0.0