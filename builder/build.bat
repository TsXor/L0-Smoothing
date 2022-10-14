@echo off
cd %~dp0
mkdir wheel_struct >nul
mkdir wheel_struct\L0_Smoothing >nul
xcopy /e /y ..\src wheel_struct\L0_Smoothing >nul
copy /y setup.py wheel_struct\setup.py >nul
copy /y ..\README.md wheel_struct\README.md >nul
cd wheel_struct
python setup.py sdist bdist_wheel
cd ..
del /s /q dist >nul
rmdir /s /q dist >nul
move /y wheel_struct\dist . >nul
del /s /q wheel_struct >nul
rmdir /s /q wheel_struct >nul
pause