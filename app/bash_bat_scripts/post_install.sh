#!/bin/bash
set -e
echo "Running post_install" > "$PREFIX/menuinst_debug.log"

"$PREFIX/bin/python" -m pip install -r "$PREFIX/PROJECT_NAME/requirements.txt" >> "$PREFIX/menuinst_debug.log"

# Check if the running platform is macOS or Linux and install additional requirements if the corresponding file exists
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS platform" >> "$PREFIX/menuinst_debug.log"
        
    if [ -f "$PREFIX/PROJECT_NAME/requirements-macos.txt" ]; then
        "$PREFIX/bin/python" -m pip install -r "$PREFIX/PROJECT_NAME/requirements-macos.txt" >> "$PREFIX/menuinst_debug.log"
    fi

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux platform" >> "$PREFIX/menuinst_debug.log"

    if [ -f "$PREFIX/PROJECT_NAME/requirements-linux.txt" ]; then
        "$PREFIX/bin/python" -m pip install -r "$PREFIX/PROJECT_NAME/requirements-linux.txt" >> "$PREFIX/menuinst_debug.log"
    fi

else
    echo "Unknown platform: $OSTYPE" >> "$PREFIX/menuinst_debug.log"
fi

PROJECT_ROOT="$PREFIX/PROJECT_NAME"
if [ -f "$PROJECT_ROOT/setup.py" ]; then
    echo "Found setup.py, installing PROJECT_NAME package locally" >> "$PREFIX/menuinst_debug.log" 
    "$PREFIX/bin/python" -m pip install "$PROJECT_ROOT" >> "$PREFIX/menuinst_debug.log"
else
    echo "No setup.py detected, skipping local pip install" >> "$PREFIX/menuinst_debug.log"
fi
"$PREFIX/bin/python" "$PREFIX/PROJECT_NAME/include_path.py" --path "$PREFIX" --files "$PREFIX/PROJECT_NAME/notebook_launcher.json" --keyword "BASE_PATH_KEYWORD" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" "$PREFIX/PROJECT_NAME/include_path.py" --path "$PREFIX" --files "$PREFIX/pre_uninstall.sh" --keyword "BASE_PATH" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" "$PREFIX/PROJECT_NAME/include_path.py" --path "$PREFIX" --files "$PREFIX/uninstall.sh" --keyword "BASE_PATH" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" "$PREFIX/PROJECT_NAME/hide_code_cells.py" "$PREFIX/PROJECT_NAME" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" -c "import os, sys; print('Python:', sys.executable); print('Prefix:', os.environ.get('PREFIX'))" >> "$PREFIX/menuinst_debug.log"
"$PREFIX/bin/python" -c "from menuinst.api import install; import os; print(install(os.path.join('$PREFIX', 'PROJECT_NAME', 'notebook_launcher.json')))" >> "$PREFIX/menuinst_debug.log" 2>&1

if [ -t 0 ]; then
    echo
    read -rp "Press Enter to close the installer..." _
fi