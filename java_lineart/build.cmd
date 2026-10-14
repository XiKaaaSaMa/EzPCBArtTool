@echo off
REM Build the filter tool. `--release 8` keeps the class runnable on a plain
REM JRE 8+ -- the target machine does not need a JDK.
REM Keep this file ASCII-only: cmd reads batch files in the OEM codepage
REM before `chcp` takes effect, and a UTF-8 comment breaks the parse.
chcp 65001 >nul
cd /d %~dp0
if not exist classes mkdir classes
javac --release 8 -Xlint:-options -encoding UTF-8 -cp "lib/*" -d classes src\LineArtJavaTool.java
if %errorlevel%==0 (echo BUILD OK) else (echo BUILD FAILED)
