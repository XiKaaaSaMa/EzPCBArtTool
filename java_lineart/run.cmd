@echo off
chcp 65001 >nul
cd /d %~dp0
java -cp "classes;lib/*" LineArtJavaTool %*
