@echo off
REM Generates the remaining Flutter platform files (existing files are kept) and fetches packages.
cd /d "%~dp0"
call flutter create --org com.faceguard --project-name face_guard --platforms android .
if errorlevel 1 exit /b 1
call flutter pub get
if errorlevel 1 exit /b 1
echo Done. Connect the phone (USB debugging on) and run:  flutter run --release
