#!/bin/bash
set -e
echo "Running post_install" > "$PREFIX/menuinst_debug.log"
BASE_REQUIREMENTS="$PREFIX/PROJECT_NAME/requirements.txt"
GPU_REQUIREMENTS="$PREFIX/PROJECT_NAME/requirements_gpu.txt"
SELECTED_REQUIREMENTS="$BASE_REQUIREMENTS"

if [ -f "$GPU_REQUIREMENTS" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS detected, installing CPU requirements from $BASE_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA GPU detected, installing GPU requirements from $GPU_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"
        SELECTED_REQUIREMENTS="$GPU_REQUIREMENTS"
    else
        echo "NVIDIA GPU not detected, installing CPU requirements from $BASE_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"
    fi
else
    echo "GPU requirements file not found, installing CPU requirements from $BASE_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"
fi


"$PREFIX/bin/python" -m pip install -r "$SELECTED_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"

# Check if the running platform is macOS or Linux and install additional requirements if the corresponding file exists
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS platform" >> "$PREFIX/menuinst_debug.log"
        
    if [ -f "$PREFIX/NucleiSky/requirements-macos.txt" ]; then
        "$PREFIX/bin/python" -m pip install -r "$PREFIX/NucleiSky/requirements-macos.txt" >> "$PREFIX/menuinst_debug.log"
    fi

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux platform" >> "$PREFIX/menuinst_debug.log"

    if [ -f "$PREFIX/NucleiSky/requirements-linux.txt" ]; then
        "$PREFIX/bin/python" -m pip install -r "$PREFIX/NucleiSky/requirements-linux.txt" >> "$PREFIX/menuinst_debug.log"
    fi

else
    echo "Unknown platform: $OSTYPE" >> "$PREFIX/menuinst_debug.log"
fi

PROJECT_ROOT="$PREFIX/NucleiSky"
if [ -f "$PROJECT_ROOT/setup.py" ]; then
    echo "Found setup.py, installing NucleiSky package locally" >> "$PREFIX/menuinst_debug.log" 
    "$PREFIX/bin/python" -m pip install "$PROJECT_ROOT" >> "$PREFIX/menuinst_debug.log"
else
    echo "No setup.py detected, skipping local pip install" >> "$PREFIX/menuinst_debug.log"
fi
"$PREFIX/bin/python" "$PREFIX/NucleiSky/include_path.py" --path "$PREFIX" --files "$PREFIX/NucleiSky/notebook_launcher.json" --keyword "BASE_PATH_KEYWORD" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" "$PREFIX/NucleiSky/include_path.py" --path "$PREFIX" --files "$PREFIX/pre_uninstall.sh" --keyword "BASE_PATH" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" "$PREFIX/NucleiSky/include_path.py" --path "$PREFIX" --files "$PREFIX/uninstall.sh" --keyword "BASE_PATH" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" "$PREFIX/NucleiSky/hide_code_cells.py" "$PREFIX/NucleiSky" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" -c "import os, sys; print('Python:', sys.executable); print('Prefix:', os.environ.get('PREFIX'))" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" -c "from menuinst.api import install; import os; print(install(os.path.join('$PREFIX', 'NucleiSky', 'notebook_launcher.json')))" >> "$PREFIX/menuinst_debug.log" 2>&1

if [ -t 0 ]; then
    echo
    read -rp "Press Enter to close the installer..." _
fi