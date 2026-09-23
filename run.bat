@echo off
REM Locate the Python interpreter installed for this project, falling back to
REM whatever `py` or `python` resolves to.
setlocal

set "PY="
for %%P in (
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "C:\Python312\python.exe"
  "C:\Python311\python.exe"
) do (
  if exist %%~P set "PY=%%~P"
)

if not defined PY (
  where py >nul 2>nul && set "PY=py"
)
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Could not find a Python interpreter.
  echo Install Python 3.11+ or set PY manually.
  REM Pause so a double-click does not hide the message.
  pause
  exit /b 1
)

REM Prefer the windowed interpreter for the GUI. Launching with python.exe
REM would keep a console window attached for as long as the app runs.
set "PYW="
for %%D in ("%PY%") do (
  if exist "%%~dpDpythonw.exe" set "PYW=%%~dpDpythonw.exe"
)
if not defined PYW set "PYW=%PY%"

set "CMD=%~1"
REM No argument means "start the application": a double-click on this file
REM should open the program, not print help and vanish.
if "%CMD%"=="" set "CMD=gui"

REM Do not `shift` here: the other branches below read %1..%8, and shifting
REM would leave them all one argument out.

if /i "%CMD%"=="gui" (
  REM `start` options must come before the title, otherwise /D is taken as the
  REM program to run and Windows reports it cannot execute the file.
  start "PCB Art Board" /D "%~dp0" "%PYW%" "%~dp0tools\gui_mockup.py" --open-preprocess
) else if /i "%CMD%"=="gui-main" (
  start "PCB Art Board" /D "%~dp0" "%PYW%" "%~dp0tools\gui_mockup.py"
) else if /i "%CMD%"=="test" (
  "%PY%" -m pytest tests -q
) else if /i "%CMD%"=="sample" (
  "%PY%" tools\make_m0_sample.py --outdir out
) else if /i "%CMD%"=="star" (
  "%PY%" tools\make_star.py --outdir out %2 %3 %4 %5 %6 %7
) else if /i "%CMD%"=="outline" (
  "%PY%" tools\make_outline_from_image.py --outdir out %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%CMD%"=="calibrate" (
  "%PY%" tools\m0_calibrate.py %2 %3 %4 %5
) else if /i "%CMD%"=="analyze" (
  node tools\analyze_epru.mjs %2
) else (
  echo Usage:
  echo   run.bat                       start the application ^(same as "gui"^)
  echo   run.bat gui                   start with the preprocessing panel open
  echo   run.bat gui-main              start with the main window only
  echo   run.bat test                  run the test suite
  echo   run.bat sample                generate M0 validation files into out\
  echo   run.bat star [--edge N]       generate a 5-point star board outline
  echo   run.bat outline ^<image^>       generate a board outline from an image
  echo   run.bat calibrate ^<file.epro2^>  calibrate units from a real project
  echo   run.bat analyze   ^<file.epru^>   structural dump of a log file
  pause
  exit /b 1
)

endlocal
