#!/bin/bash
# ---------------------------------------------------------------------------
# build_cellmatchr_app.sh — rebuild CellMatchR.app from scratch.
#
# WHY THIS DESIGN:
#   Under macOS Tahoe (26+), an app launched from Finder runs in a sandbox that
#   is DENIED read access to protected folders (Desktop, Documents, Downloads).
#   This project lives on the Desktop, so a Finder-launched app gets
#   "Operation not permitted" when its Python tries to read the project code.
#   Terminal, however, IS permitted to read the Desktop. So this app is a thin
#   launcher: it hands off to scripts/launch_cellmatchr.command via Terminal,
#   which then runs the real GUI with Terminal's file-access permission.
#   The app itself never touches the Desktop filesystem (it only passes a path
#   string to Terminal), so it is immune to the TCC restriction.
#
# USAGE:  bash scripts/build_cellmatchr_app.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
APP="$ROOT_DIR/CellMatchR.app"
CONTENTS="$APP/Contents"

echo "Project root : $ROOT_DIR"
echo "Building     : $APP"

# 1) Fresh bundle skeleton --------------------------------------------------
rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

# 2) Info.plist -------------------------------------------------------------
cat > "$CONTENTS/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>            <string>CellMatchR</string>
    <key>CFBundleDisplayName</key>     <string>CellMatchR</string>
    <key>CFBundleIdentifier</key>      <string>com.cellmatchr.launcher</string>
    <key>CFBundleVersion</key>         <string>0.1.0</string>
    <key>CFBundleShortVersionString</key> <string>0.1.0</string>
    <key>CFBundlePackageType</key>     <string>APPL</string>
    <key>CFBundleExecutable</key>      <string>CellMatchR</string>
    <key>CFBundleIconFile</key>        <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>  <string>10.13</string>
    <key>NSHighResolutionCapable</key> <true/>
    <key>LSBackgroundOnly</key>        <false/>
</dict>
</plist>
PLIST

# 3) Main executable: thin Terminal-handoff launcher ------------------------
cat > "$CONTENTS/MacOS/CellMatchR" <<'LAUNCH'
#!/bin/bash
# CellMatchR.app — thin launcher (see build_cellmatchr_app.sh for rationale).
# Resolve the project root from our own bundle path using STRING math only —
# no filesystem access, so the Tahoe Desktop restriction never applies to us.
SELF="$0"                                  # .../CellMatchR.app/Contents/MacOS/CellMatchR
ROOT="${SELF%/CellMatchR.app/*}"           # .../<project dir>
CMD="$ROOT/scripts/launch_cellmatchr.command"

# Hand off to Terminal, which IS allowed to read the Desktop. Terminal runs the
# .command (and the Python GUI it starts) with its own file-access permission.
osascript >/dev/null 2>&1 <<OSA
tell application "Terminal"
    activate
    do script "'$CMD'"
end tell
OSA
LAUNCH
chmod +x "$CONTENTS/MacOS/CellMatchR"

# 4) Icon -------------------------------------------------------------------
if [ -f "$SCRIPT_DIR/AppIcon.icns" ]; then
    cp "$SCRIPT_DIR/AppIcon.icns" "$CONTENTS/Resources/AppIcon.icns"
else
    echo "WARNING: scripts/AppIcon.icns not found — app will use a generic icon."
fi

# 5) Clean metadata + ad-hoc sign (keeps Gatekeeper quiet; not required to run)
xattr -cr "$APP" 2>/dev/null || true
codesign --force --sign - "$APP" 2>/dev/null || true

echo "Done. Built $APP"
