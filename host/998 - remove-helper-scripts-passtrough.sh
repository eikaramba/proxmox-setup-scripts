#!/usr/bin/env bash
#
# Proxmox LXC GPU Passthrough Cleanup Script
#
# Removes ALL GPU/DRI/KFD passthrough lines from an existing LXC container
# config. Useful for stripping the auto-configured passthrough added by
# community helper-scripts before applying a custom passthrough setup.
#
# Usage: ./998-remove-helper-scripts-passthrough.sh <CONTAINER_ID>
#

set -euo pipefail

err()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }
warn() { echo "WARN: $*" >&2; }

[ "$(id -u)" -eq 0 ] || err "This script must be run as root."
[ $# -eq 1 ] || err "Usage: $0 <CONTAINER_ID>"

CONTAINER_ID="$1"
CONFIG_FILE="/etc/pve/lxc/${CONTAINER_ID}.conf"

[ -f "$CONFIG_FILE" ] || err "Container ${CONTAINER_ID} not found at ${CONFIG_FILE}."
command -v pct >/dev/null 2>&1 || err "'pct' not found. Are you on a Proxmox host?"

# ---------------------------------------------------------------------------
# Stop container
# ---------------------------------------------------------------------------

info "Stopping container ${CONTAINER_ID} (if running)..."
pct stop "$CONTAINER_ID" >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
    [ "$(pct status "$CONTAINER_ID" | awk '{print $2}')" = "stopped" ] && break
    sleep 1
done
[ "$(pct status "$CONTAINER_ID" | awk '{print $2}')" = "stopped" ] \
    || err "Container did not stop within 30s."

# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

BACKUP_FILE="${CONFIG_FILE}.backup-cleanup-$(date +%Y%m%d-%H%M%S)"
cp -a "$CONFIG_FILE" "$BACKUP_FILE"
info "Backup written to: ${BACKUP_FILE}"

# ---------------------------------------------------------------------------
# Show what will be removed
# ---------------------------------------------------------------------------

info "GPU-related lines currently in config:"
grep -nE \
    -e '^\s*dev[0-9]+:\s*/dev/(dri|kfd)' \
    -e '^\s*lxc\.cgroup2\.devices\.allow:\s*c\s+(226|254|509):' \
    -e '^\s*lxc\.mount\.entry:.*dev/(dri|kfd)' \
    -e '^\s*lxc\.apparmor\.profile:\s*unconfined' \
    -e '^\s*lxc\.cap\.drop:\s*$' \
    -e '^\s*#\s*---\s*AMD GPU Passthrough' \
    -e '^\s*#\s*---\s*end AMD GPU Passthrough' \
    "$CONFIG_FILE" || info "  (none found)"

read -rp "Remove these lines? [y/N]: " CONT
[[ "$CONT" =~ ^[Yy]$ ]] || { info "Aborted; config unchanged."; exit 0; }

# ---------------------------------------------------------------------------
# Strip GPU lines
#
# We remove:
#   - dev0:/dev/dri/... and dev1:/dev/kfd style passthrough entries
#     (helper-scripts uses these via `pct set --devN`)
#   - lxc.cgroup2.devices.allow lines for DRM (major 226) and KFD
#     (KFD major is dynamic; common values include 254, 509, etc.,
#      so we match by referenced device path where possible)
#   - lxc.mount.entry lines targeting dev/dri or dev/kfd
#   - lxc.apparmor.profile: unconfined
#   - lxc.cap.drop: (empty value)
#   - the marker comments from our own upgrade script
# ---------------------------------------------------------------------------

TMP_FILE=$(mktemp)

# Build the major numbers actually in use on the host so we don't accidentally
# leave a stale cgroup line referencing the GPU.
HOST_MAJORS=()
if [ -e /dev/dri ]; then
    while read -r maj; do HOST_MAJORS+=("$maj"); done < <(
        ls -al /dev/dri/card* /dev/dri/renderD* 2>/dev/null \
        | sed 's/,//' | awk '{print $5}' | sort -u
    )
fi
if [ -e /dev/kfd ]; then
    KFD_MAJ=$(ls -al /dev/kfd | sed 's/,//' | awk '{print $5}')
    [ -n "$KFD_MAJ" ] && HOST_MAJORS+=("$KFD_MAJ")
fi

# Always include 226 (standard DRM) as a safety net
HOST_MAJORS+=("226")

# Build alternation regex of majors
MAJ_RE=$(printf '%s|' "${HOST_MAJORS[@]}" | sed 's/|$//')

awk -v majre="$MAJ_RE" '
    # dev0:/dev/dri/... or dev1:/dev/kfd
    /^[[:space:]]*dev[0-9]+:[[:space:]]*\/dev\/(dri|kfd)/ { next }

    # lxc.cgroup2.devices.allow for known GPU/KFD majors
    /^[[:space:]]*lxc\.cgroup2\.devices\.allow:[[:space:]]*c[[:space:]]+/ {
        if (match($0, "c[[:space:]]+(" majre "):")) next
    }

    # lxc.mount.entry referencing dev/dri or dev/kfd
    /^[[:space:]]*lxc\.mount\.entry:.*dev\/(dri|kfd)/ { next }

    # AppArmor + cap drop lines (added by GPU passthrough scripts)
    /^[[:space:]]*lxc\.apparmor\.profile:[[:space:]]*unconfined[[:space:]]*$/ { next }
    /^[[:space:]]*lxc\.cap\.drop:[[:space:]]*$/ { next }

    # Marker comments from our upgrade script
    /^[[:space:]]*#[[:space:]]*---[[:space:]]*AMD GPU Passthrough/ { next }
    /^[[:space:]]*#[[:space:]]*---[[:space:]]*end AMD GPU Passthrough/ { next }

    { print }
' "$CONFIG_FILE" > "$TMP_FILE"

# Collapse multiple consecutive blank lines into one for tidiness
awk 'NF || prev { print } { prev = NF }' "$TMP_FILE" > "${TMP_FILE}.2"
mv "${TMP_FILE}.2" "$TMP_FILE"

cat "$TMP_FILE" > "$CONFIG_FILE"
rm -f "$TMP_FILE"

info "GPU passthrough lines removed."
info "You can now run the upgrade script to apply a clean GPU configuration:"
echo "    ./032 - upgrade-lxc-gpu.sh ${CONTAINER_ID}"
info "If anything went wrong, restore with:"
echo "    cp -a ${BACKUP_FILE} ${CONFIG_FILE}"
