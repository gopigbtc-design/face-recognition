#!/usr/bin/env bash
# Generates the remaining Flutter platform files (existing files are kept) and fetches packages.
set -e
cd "$(dirname "$0")"
flutter create --org com.faceguard --project-name face_guard --platforms android .
flutter pub get
echo "Done. Connect the phone (USB debugging on) and run:  flutter run --release"
