#!/bin/sh
# Adds Fatwa Transcript to your applications menu. No root needed. Run it from the unpacked folder:
#   ./install.sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS"
cat > "$APPS/fatwa-transcript.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Fatwa Transcript
Comment=Arabic voice notes to corrected text
Exec="$DIR/FatwaTranscript"
Terminal=false
Categories=AudioVideo;Audio;Office;
EOF
chmod +x "$DIR/FatwaTranscript"
echo "Added Fatwa Transcript to your applications menu."
echo "Keep this folder where it is: $DIR"
