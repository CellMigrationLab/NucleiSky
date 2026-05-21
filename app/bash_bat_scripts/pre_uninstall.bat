@ECHO ON
echo Running pre-uninstall
"%PREFIX%\python.exe" -c "from menuinst.api import remove; import os; remove(os.path.join(r'%PREFIX%', 'PROJECT_NAME', 'notebook_launcher.json'))"
SET "ARP_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\UNDERSCORED_PROJECT_NAME"
reg delete "%ARP_KEY%" /f >NUL 2>&1
echo Pre-uninstall completed!
SetLocal EnableDelayedExpansion
