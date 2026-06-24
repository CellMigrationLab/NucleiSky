@ECHO OFF
SETLOCAL
echo Running post_install > "%PREFIX%\menuinst_debug.log"
SET "BASE_REQUIREMENTS=%PREFIX%\PROJECT_NAME\requirements.txt"
SET "GPU_REQUIREMENTS=%PREFIX%\PROJECT_NAME\requirements_gpu.txt"
SET "SELECTED_REQUIREMENTS=%BASE_REQUIREMENTS%"

IF EXIST "%GPU_REQUIREMENTS%" (
    where nvidia-smi >NUL 2>&1
    IF ERRORLEVEL 1 (
        echo NVIDIA GPU not detected, installing CPU requirements from "%BASE_REQUIREMENTS%" >> "%PREFIX%\menuinst_debug.log"
    ) ELSE (
        echo NVIDIA GPU detected, installing GPU requirements from "%GPU_REQUIREMENTS%" >> "%PREFIX%\menuinst_debug.log"
        SET "SELECTED_REQUIREMENTS=%GPU_REQUIREMENTS%"
    )
) ELSE (
    echo GPU requirements file not found, installing CPU requirements from "%BASE_REQUIREMENTS%" >> "%PREFIX%\menuinst_debug.log"
)

"%PREFIX%\python.exe" -m pip install -r "%SELECTED_REQUIREMENTS%" >> "%PREFIX%\menuinst_debug.log"

IF EXIST "%PREFIX%\NucleiSky\requirements-windows.txt" (
    "%PREFIX%\python.exe" -m pip install -r "%PREFIX%\NucleiSky\requirements-windows.txt" >> "%PREFIX%\menuinst_debug.log"
)

SET "PROJECT_ROOT=%PREFIX%\NucleiSky"
IF EXIST "%PROJECT_ROOT%\setup.py" (
    echo Found setup.py, installing NucleiSky package locally >> "%PREFIX%\menuinst_debug.log"
    "%PREFIX%\python.exe" -m pip install "%PROJECT_ROOT%" >> "%PREFIX%\menuinst_debug.log"
) ELSE (
    echo No setup.py detected, skipping local pip install >> "%PREFIX%\menuinst_debug.log"
)
"%PREFIX%\python.exe" "%PREFIX%\NucleiSky\include_path.py" --path "%PREFIX%" --files "%PREFIX%\NucleiSky\notebook_launcher.json" --keyword "BASE_PATH_KEYWORD" >> "%PREFIX%\menuinst_debug.log"
"%PREFIX%\python.exe" "%PREFIX%\NucleiSky\hide_code_cells.py" "%PREFIX%\NucleiSky" >> "%PREFIX%\menuinst_debug.log"
"%PREFIX%\python.exe" -c "import os, sys; print('Python:', sys.executable); print('Prefix:', os.environ.get('PREFIX'))" >> "%PREFIX%\menuinst_debug.log"
"%PREFIX%\python.exe" -c "from menuinst.api import install; import os; print(install(os.path.join(r'%PREFIX%', 'NucleiSky', 'notebook_launcher.json')))" >> "%PREFIX%\menuinst_debug.log" 2>&1

SET "ARP_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\NucleiSky"
SET "UNINSTALL_EXE=%PREFIX%\Uninstall-NucleiSky.exe"
SET "DISPLAY_ICON=%PREFIX%\NucleiSky\logo.ico"
SET "DISPLAY_VERSION=0.0.1"
SET "PUBLISHER=CellMigrationLab"
echo Registering NucleiSky in Windows Apps list >> "%PREFIX%\menuinst_debug.log"
reg add "%ARP_KEY%" /v DisplayName /d "NucleiSky" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v DisplayVersion /d "%DISPLAY_VERSION%" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v Publisher /d "%PUBLISHER%" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v InstallLocation /d "%PREFIX%" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v DisplayIcon /d "%DISPLAY_ICON%" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v UninstallString /d "\"%UNINSTALL_EXE%\"" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v QuietUninstallString /d "\"%UNINSTALL_EXE%\" /S" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v NoModify /t REG_DWORD /d 1 /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v NoRepair /t REG_DWORD /d 1 /f >> "%PREFIX%\menuinst_debug.log" 2>&1

echo Post-install completed!
ENDLOCAL
