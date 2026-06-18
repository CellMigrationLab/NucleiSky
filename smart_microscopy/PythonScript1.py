# IMPORTANT: 'limjob' must be imported like this (not from nor as)
import limjob

from datetime import datetime
from pathlib import Path
import shutil
import time

def run(imgs: tuple[limjob.Image], Job: limjob.JobParam, macro: limjob.MacroParam, ctx: limjob.RunContext):
    try:
        today = datetime.now().strftime("%Y%m%d")
        output_path = r"G:"
        output_dir = Path(output_path) / f"{today}_Experiment" / "local_logging"
        save_dir = Path(output_path) / f"{today}_Experiment" / f"{datetime.now().strftime('%H%M%S')}"
        shutil.move(output_dir, save_dir)
    except Exception as e:
        time.sleep(5)
        run(imgs, Job, macro, ctx)