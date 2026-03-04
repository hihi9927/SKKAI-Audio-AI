@echo off
set CLASSPATH=
set JAVA_HOME=C:\Program Files\Java\jdk-17.0.2
cd /d D:\STiTy-main\STiTy-Mobile\android
echo Current dir: %CD%
call "D:\STiTy-main\STiTy-Mobile\android\gradlew.bat" assembleRelease
