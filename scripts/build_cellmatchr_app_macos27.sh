#!/bin/bash
# ---------------------------------------------------------------------------
# build_cellmatchr_app_macos27.sh — build "CellMatchR (macOS 27+).app".
#
# Use THIS app on macOS 27 (Golden Gate) and later. On macOS 26 (Tahoe) and
# earlier, keep using the original CellMatchR.app (scripts/build_cellmatchr_app.sh),
# which is left unchanged.
#
# WHY A SEPARATE APP:
#   The original CellMatchR.app is a bash script that asks Terminal (via an
#   AppleScript `do script`) to run scripts/launch_cellmatchr.command, so the
#   GUI runs with Terminal's permission to read the Desktop. macOS 27 no longer
#   allows that: a command another app asks Terminal to run is checked against
#   the *requesting* app's permissions, not Terminal's. A bash-script app is
#   seen as /bin/bash, a system program macOS will not show a permission
#   prompt for, so Desktop access is silently denied ("Operation not
#   permitted") in the Terminal window it opens.
#
#   This app is instead a compiled AppleScript applet with its own identity,
#   which macOS CAN prompt for. It hands off with `open -a Terminal`, the same
#   way Finder opens a double-clicked .command file, so no "control Terminal"
#   (Automation) permission is needed. On first launch macOS may ask to allow
#   "CellMatchR (macOS 27+)" access to the Desktop folder — click Allow.
#
#   LSMinimumSystemVersion is 27.0, so on older macOS this app refuses to
#   open with a clear "needs a newer macOS" message instead of failing oddly.
#
# USAGE:  bash scripts/build_cellmatchr_app_macos27.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
APP_NAME="CellMatchR (macOS 27+)"
APP="$ROOT_DIR/$APP_NAME.app"
PLIST="$APP/Contents/Info.plist"

echo "Project root : $ROOT_DIR"
echo "Building     : $APP"

# 1) Compile the launcher applet ---------------------------------------------
# The project root is derived from the app's own path with string math only;
# the app must stay inside the CellMatchR folder, next to scripts/.
rm -rf "$APP"
/usr/bin/osacompile -o "$APP" <<'APPLESCRIPT'
-- CellMatchR launcher for macOS 27 (Golden Gate) and later.
-- See scripts/build_cellmatchr_app_macos27.sh for why this exists.
on run
	set appPath to POSIX path of (path to me) -- ".../<project>/CellMatchR (macOS 27+).app/"
	if appPath ends with "/" then set appPath to text 1 thru -2 of appPath
	set oldDelims to AppleScript's text item delimiters
	set AppleScript's text item delimiters to "/"
	set projectRoot to (text items 1 thru -2 of appPath) as text
	set AppleScript's text item delimiters to oldDelims
	set cmdPath to projectRoot & "/scripts/launch_cellmatchr.command"
	try
		do shell script "/usr/bin/open -a Terminal " & quoted form of cmdPath
	on error errMsg
		display alert "CellMatchR could not start" message ¬
			("Could not open:" & return & cmdPath & return & return & ¬
			"Keep this app inside the CellMatchR folder (next to the scripts folder). " & ¬
			"To launch from the Dock, use an alias instead of moving the app." & return & return & ¬
			"Details: " & errMsg) as critical
	end try
end run
APPLESCRIPT

# 2) Info.plist: identity + macOS 27 minimum ----------------------------------
set_key() {  # set_key <key> <type> <value> — Set if present, otherwise Add
    /usr/libexec/PlistBuddy -c "Set :$1 $3" "$PLIST" 2>/dev/null \
        || /usr/libexec/PlistBuddy -c "Add :$1 $2 $3" "$PLIST"
}
set_key CFBundleIdentifier         string "com.cellmatchr.launcher.macos27"
set_key CFBundleName               string "CellMatchR"
set_key CFBundleDisplayName        string "$APP_NAME"
set_key CFBundleShortVersionString string "0.1.0"
set_key CFBundleVersion            string "0.1.0"
set_key LSMinimumSystemVersion     string "27.0"
# osacompile also writes a per-architecture minimum; drop it so the 27.0 floor above is the one macOS uses.
/usr/libexec/PlistBuddy -c "Delete :LSMinimumSystemVersionByArchitecture" "$PLIST" 2>/dev/null || true

# 3) Icon: osacompile points CFBundleIconFile at applet.icns — replace it ------
# It also adds an asset catalog (Assets.car + CFBundleIconName) holding the
# default script icon, which modern macOS prefers over CFBundleIconFile; remove
# both so the CellMatchR icon is the one shown.
if [ -f "$SCRIPT_DIR/AppIcon.icns" ]; then
    cp "$SCRIPT_DIR/AppIcon.icns" "$APP/Contents/Resources/applet.icns"
    rm -f "$APP/Contents/Resources/Assets.car"
    /usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$PLIST" 2>/dev/null || true
else
    echo "WARNING: scripts/AppIcon.icns not found — app will use the default applet icon."
fi

# 4) Ad-hoc sign (required after editing Info.plist / resources) --------------
xattr -cr "$APP"
codesign --force --deep --sign - "$APP"

echo ""
echo "Built: $APP"
echo "First launch: if macOS asks to allow Desktop access for \"$APP_NAME\", click Allow."
