@echo off
setlocal enableDelayedExpansion

cd /D "%~dp0"

SET MAMBA_ROOT_PREFIX=%cd%\.micromamba\windows
SET RELEASE_URL=https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64

REM check installation

:check
IF NOT EXIST "%MAMBA_ROOT_PREFIX%" goto install
call %MAMBA_ROOT_PREFIX%\micromamba.exe shell hook -s cmd.exe -p "%MAMBA_ROOT_PREFIX%"
if %errorlevel% NEQ 0 goto install
call "%MAMBA_ROOT_PREFIX%\condabin\mamba_hook.bat"
if %errorlevel% NEQ 0 goto install
call micromamba activate Moru
if %errorlevel% NEQ 0 goto install
goto menu

:install
cls
REM echo Downloading micromamba from %RELEASE_URL%
curl.exe -L -o micromamba.exe "%RELEASE_URL%"
IF NOT EXIST "%MAMBA_ROOT_PREFIX%" mkdir "%MAMBA_ROOT_PREFIX%" 2>nul
REM echo Installing micromamba to %MAMBA_INSTALL_PATH%
move /Y micromamba.exe "%MAMBA_ROOT_PREFIX%\micromamba.exe"
echo Installing Moru...
call micromamba create -f env-win.yml -y
call micromamba clean -a -f -y
call micromamba activate Moru
call pip cache purge
goto check

:menu
cls
echo.
echo Please select an option:
echo 1) Start Trainer
echo 2) Update Moru
echo 3) Open CMD
echo 4) Exit

echo.

set /p userchoice=Enter your choice (1 ~ 4):

if "%userchoice%"=="1" goto start
if "%userchoice%"=="2" goto update
if "%userchoice%"=="3" goto cmd
if "%userchoice%"=="4" exit

echo Invalid choice
pause
goto menu

:start
call python server.py
exit

:update
echo Updating Moru...
call md5.bat "%~f0" hash1
call git pull --autostash
call md5.bat "%~f0" hash2
IF "!hash1!" NEQ "!hash2!" (
	echo "start_windows.bat has been updated. Restarting..."
	REM "update"
	pause
	start "" "%~f0"
	exit /b
)

call micromamba update -f env-win.yml -y
call micromamba clean -a -f -y
call pip cache purge
start "" "%~f0"
call exit /b

:cmd
cls
echo Type 'exit' to return to menu.
cmd /k
goto menu

:end
pause