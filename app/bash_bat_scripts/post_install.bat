@ECHO OFF
echo Running post_install > "%PREFIX%\menuinst_debug.log"
"%PREFIX%\python.exe" -m pip install -r "%PREFIX%\PROJECT_NAME\requirements.txt" >> "%PREFIX%\menuinst_debug.log"

IF EXIST "%PREFIX%\PROJECT_NAME\requirements-windows.txt" (
    "%PREFIX%\python.exe" -m pip install -r "%PREFIX%\PROJECT_NAME\requirements-windows.txt" >> "%PREFIX%\menuinst_debug.log"
)

SET "PROJECT_ROOT=%PREFIX%\PROJECT_NAME"
IF EXIST "%PROJECT_ROOT%\setup.py" (
    echo Found setup.py, installing PROJECT_NAME package locally >> "%PREFIX%\menuinst_debug.log"
    "%PREFIX%\python.exe" -m pip install "%PROJECT_ROOT%" >> "%PREFIX%\menuinst_debug.log"
) ELSE (
    echo No setup.py detected, skipping local pip install >> "%PREFIX%\menuinst_debug.log"
)
"%PREFIX%\python.exe" "%PREFIX%\PROJECT_NAME\include_path.py" --path "%PREFIX%" --files "%PREFIX%\PROJECT_NAME\notebook_launcher.json" --keyword "BASE_PATH_KEYWORD" >> "%PREFIX%\menuinst_debug.log"
"%PREFIX%\python.exe" "%PREFIX%\PROJECT_NAME\hide_code_cells.py" "%PREFIX%\PROJECT_NAME" >> "%PREFIX%\menuinst_debug.log"
"%PREFIX%\python.exe" -c "import os, sys; print('Python:', sys.executable); print('Prefix:', os.environ.get('PREFIX'))" >> "%PREFIX%\menuinst_debug.log"
"%PREFIX%\python.exe" -c "from menuinst.api import install; import os; print(install(os.path.join(r'%PREFIX%', 'PROJECT_NAME', 'notebook_launcher.json')))" >> "%PREFIX%\menuinst_debug.log" 2>&1

SET "ARP_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\UNDERSCORED_PROJECT_NAME"
SET "UNINSTALL_EXE=%PREFIX%\Uninstall-UNDERSCORED_PROJECT_NAME.exe"
SET "DISPLAY_ICON=%PREFIX%\PROJECT_NAME\logo.ico"
SET "DISPLAY_VERSION=VERSION_NUMBER"
SET "PUBLISHER=GITHUB_OWNER"
echo Registering PROJECT_NAME in Windows Apps list >> "%PREFIX%\menuinst_debug.log"
reg add "%ARP_KEY%" /v DisplayName /d "PROJECT_NAME" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v DisplayVersion /d "%DISPLAY_VERSION%" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v Publisher /d "%PUBLISHER%" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v InstallLocation /d "%PREFIX%" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v DisplayIcon /d "%DISPLAY_ICON%" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v UninstallString /d "\"%UNINSTALL_EXE%\"" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v QuietUninstallString /d "\"%UNINSTALL_EXE%\" /S" /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v NoModify /t REG_DWORD /d 1 /f >> "%PREFIX%\menuinst_debug.log" 2>&1
reg add "%ARP_KEY%" /v NoRepair /t REG_DWORD /d 1 /f >> "%PREFIX%\menuinst_debug.log" 2>&1

echo Post-install completed!
SetLocal EnableDelayedExpansion
