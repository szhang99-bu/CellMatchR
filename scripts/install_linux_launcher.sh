#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
    printf '%s\n' 'This installer is for Linux.' >&2
    exit 1
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${XDG_DATA_HOME:-$HOME/.local/share}"
if [[ "$data_dir" != /* ]]; then
    data_dir="$HOME/.local/share"
fi

desktop_value() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value// /\\s}"
    value="${value//$'\t'/\\t}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    printf '%s' "$value"
}

applications="$data_dir/applications"
desktop_file="$applications/com.cellmatchr.launcher.desktop"
mkdir -p -- "$applications"
pending="$(mktemp -- "$applications/.cellmatchr.XXXXXX")"
trap 'rm -f -- "$pending"' EXIT
cat > "$pending" <<EOF
[Desktop Entry]
Type=Application
Name=CellMatchR (Linux)
Comment=Cell segmentation, review, and colocalization
Exec=bash ./run-linux.sh
Path=$(desktop_value "$repo_dir")
Icon=$(desktop_value "$repo_dir/CellMatchr_Logo.png")
Terminal=false
Categories=Science;Biology;
EOF
mv -f -- "$pending" "$desktop_file"
printf 'Installed CellMatchR (Linux) in your app menu.\n%s\n' "$desktop_file"
