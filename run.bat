@echo off
setlocal
cd /d "%~dp0" || exit /b 1

REM An explicit interpreter is authoritative, including when it is invalid.
set "PY="
if not defined PCBART_PYTHON goto :discover_python
call :check_python "%PCBART_PYTHON%"
if defined PY goto :python_found
echo PCBART_PYTHON must point to a working Python 3.11 or 3.12 executable. 1>&2
exit /b 1

:discover_python

REM Prefer registered supported versions, then common installations and PATH.
where py >nul 2>nul
if not errorlevel 1 (
  for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do call :check_python "%%P"
  if not defined PY for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do call :check_python "%%P"
)
if not defined PY for %%P in (
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "C:\Python312\python.exe"
  "C:\Python311\python.exe"
) do if not defined PY call :check_python "%%~P"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY call :check_python "%%P"
if not defined PY (
  echo Could not find Python 3.11 or 3.12. Install one or set PCBART_PYTHON. 1>&2
  exit /b 1
)

:python_found
set "PYW="
for %%D in ("%PY%") do if exist "%%~dpDpythonw.exe" set "PYW=%%~dpDpythonw.exe"
if not defined PYW set "PYW=%PY%"
set "CMD=%~1"
if "%CMD%"=="" set "CMD=gui"

if /i "%CMD%"=="gui" (
  start "PCB Art Board" /D "%~dp0" "%PYW%" "%~dp0tools\gui_mockup.py" --open-preprocess
  exit /b 0
)
if /i "%CMD%"=="gui-main" (
  start "PCB Art Board" /D "%~dp0" "%PYW%" "%~dp0tools\gui_mockup.py"
  exit /b 0
)
if /i "%CMD%"=="test" goto :test
if /i "%CMD%"=="sample" (
  "%PY%" tools\make_m0_sample.py --outdir out
  goto :return_status
)
if /i "%CMD%"=="star" (
  "%PY%" tools\make_star.py --outdir out %2 %3 %4 %5 %6 %7
  goto :return_status
)
if /i "%CMD%"=="outline" (
  "%PY%" tools\make_outline_from_image.py --outdir out %2 %3 %4 %5 %6 %7 %8 %9
  goto :return_status
)
if /i "%CMD%"=="calibrate" (
  "%PY%" tools\m0_calibrate.py %2 %3 %4 %5
  goto :return_status
)
if /i "%CMD%"=="analyze" (
  node tools\analyze_epru.mjs %2
  goto :return_status
)
echo Usage:
echo   run.bat                       start the application ^(same as "gui"^)
echo   run.bat gui                   start with the preprocessing panel open
echo   run.bat gui-main              start with the main window only
echo   run.bat test                  run the test suite
echo   run.bat sample                generate M0 validation files into out\
echo   run.bat star [--edge N]       generate a 5-point star board outline
echo   run.bat outline ^<image^>       generate a board outline from an image
echo   run.bat calibrate ^<file.epro2^>  calibrate units from a real project
echo   run.bat analyze ^<file.epru^>   structural dump of a log file
exit /b 1

:test
"%PY%" -c "import pytest" >nul 2>nul
if errorlevel 1 (
  echo pytest is not installed for "%PY%". Install the project requirements. 1>&2
  exit /b 1
)
"%PY%" -m pytest tests -q
:return_status
exit /b %errorlevel%

:check_python
if not exist "%~1" exit /b 1
"%~1" -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
exit /b 0
