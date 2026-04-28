#!/usr/bin/env bash
#
# Proxmox LXC GPU Passthrough Upgrade Script (AMD / Strix Halo)
#
# Adds AMD GPU passthrough (DRI + KFD/ROCm) to an existing unprivileged or
# privileged LXC container on Proxmox VE.
#
# Usage: ./upgrade-lxc-gpu.sh <CONTAINER_ID>
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

err() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }
warn() { echo "WARN: $*" >&2; }

# ---------------------------------------------------------------------------
# Argument & environment checks
# ---------------------------------------------------------------------------

[ "$(id -u)" -eq 0 ] || err "This script must be run as root."

if [ $# -ne 1 ]; then
    err "Usage: $0 <CONTAINER_ID>"
fi

CONTAINER_ID="$1"
CONFIG_FILE="/etc/pve/lxc/${CONTAINER_ID}.conf"

[ -f "$CONFIG_FILE" ] || err "Container ${CONTAINER_ID} not found at ${CONFIG_FILE}."

command -v pct >/dev/null 2>&1 || err "'pct' not found. Are you on a Proxmox host?"

# ---------------------------------------------------------------------------
# Idempotency: refuse if the GPU block was already added
# ---------------------------------------------------------------------------

GPU_MARKER="# --- AMD GPU Passthrough (added by upgrade-lxc-gpu.sh) ---"

if grep -qF "$GPU_MARKER" "$CONFIG_FILE"; then
    err "GPU passthrough block already present in ${CONFIG_FILE}. Remove it first if you want to re-run."
fi

# ---------------------------------------------------------------------------
# Detect available AMD GPUs via /dev/dri/by-path
# ---------------------------------------------------------------------------

if [ ! -d /dev/dri/by-path ]; then
    err "/dev/dri/by-path does not exist. No DRM devices found on host."
fi

mapfile -t PCI_CARDS < <(ls /dev/dri/by-path/ 2>/dev/null | grep -E '^pci-.*-card$' | sed -E 's/^pci-(.*)-card$/\1/')

if [ "${#PCI_CARDS[@]}" -eq 0 ]; then
    err "No AMD/DRM GPUs found under /dev/dri/by-path/."
fi

info "Detected GPU PCI addresses:"
for i in "${!PCI_CARDS[@]}"; do
    PCI="${PCI_CARDS[$i]}"
    REAL=$(readlink -f "/dev/dri/by-path/pci-${PCI}-card" 2>/dev/null || echo "?")
    echo "  [$i] ${PCI}  ->  ${REAL}"
done

if [ "${#PCI_CARDS[@]}" -eq 1 ]; then
    PCI_ADDRESS="${PCI_CARDS[0]}"
    info "Using only available GPU: ${PCI_ADDRESS}"
else
    read -rp "Select GPU index [0-$((${#PCI_CARDS[@]}-1))]: " IDX
    [[ "$IDX" =~ ^[0-9]+$ ]] || err "Invalid selection."
    [ "$IDX" -lt "${#PCI_CARDS[@]}" ] || err "Index out of range."
    PCI_ADDRESS="${PCI_CARDS[$IDX]}"
fi

CARD_PATH="/dev/dri/by-path/pci-${PCI_ADDRESS}-card"
RENDER_PATH="/dev/dri/by-path/pci-${PCI_ADDRESS}-render"

[ -e "$CARD_PATH" ]   || err "Card device not found: ${CARD_PATH}"
[ -e "$RENDER_PATH" ] || err "Render device not found: ${RENDER_PATH}"

# ---------------------------------------------------------------------------
# Derive cgroup major/minor from the actually selected devices
# ---------------------------------------------------------------------------

REAL_CARD=$(readlink -f "$CARD_PATH")
REAL_RENDER=$(readlink -f "$RENDER_PATH")

read -r CARD_MAJOR CARD_MINOR < <(ls -al "$REAL_CARD"   | sed 's/,//' | awk '{print $5, $6}')
read -r REND_MAJOR REND_MINOR < <(ls -al "$REAL_RENDER" | sed 's/,//' | awk '{print $5, $6}')

[ -n "${CARD_MAJOR:-}" ] && [ -n "${CARD_MINOR:-}" ] || err "Failed to read major/minor for ${REAL_CARD}."
[ -n "${REND_MAJOR:-}" ] && [ -n "${REND_MINOR:-}" ] || err "Failed to read major/minor for ${REAL_RENDER}."

info "Card device:   ${REAL_CARD}   (cgroup ${CARD_MAJOR}:${CARD_MINOR})"
info "Render device: ${REAL_RENDER} (cgroup ${REND_MAJOR}:${REND_MINOR})"

# ---------------------------------------------------------------------------
# Detect /dev/kfd (ROCm). Optional but required for compute on Strix Halo.
# ---------------------------------------------------------------------------

HAS_KFD=0
if [ -e /dev/kfd ]; then
    read -r KFD_MAJOR KFD_MINOR < <(ls -al /dev/kfd | sed 's/,//' | awk '{print $5, $6}')
    if [ -n "${KFD_MAJOR:-}" ] && [ -n "${KFD_MINOR:-}" ]; then
        HAS_KFD=1
        info "KFD device:    /dev/kfd          (cgroup ${KFD_MAJOR}:${KFD_MINOR})"
    else
        warn "/dev/kfd exists but major/minor could not be read; skipping KFD passthrough."
    fi
else
    warn "/dev/kfd not present on host. ROCm/compute workloads will not work in the container."
fi

# ---------------------------------------------------------------------------
# Check for duplicate mount targets already in container config
# ---------------------------------------------------------------------------

if grep -qE 'lxc\.mount\.entry:.*dev/dri/card0( |$)' "$CONFIG_FILE"; then
    err "Container already has a mount entry for dev/dri/card0. Remove it first."
fi
if grep -qE 'lxc\.mount\.entry:.*dev/dri/renderD128( |$)' "$CONFIG_FILE"; then
    err "Container already has a mount entry for dev/dri/renderD128. Remove it first."
fi
if [ "$HAS_KFD" -eq 1 ] && grep -qE 'lxc\.mount\.entry:.*dev/kfd( |$)' "$CONFIG_FILE"; then
    err "Container already has a mount entry for dev/kfd. Remove it first."
fi

# ---------------------------------------------------------------------------
# Privileged / unprivileged check
# ---------------------------------------------------------------------------

UNPRIV=$(grep -E '^unprivileged:\s*1' "$CONFIG_FILE" || true)
if [ -n "$UNPRIV" ]; then
    warn "Container ${CONTAINER_ID} is UNPRIVILEGED."
    warn "GPU access will almost certainly fail without UID/GID mapping for the"
    warn "'video', 'render' (and possibly 'kfd') groups, plus matching group"
    warn "membership for the user inside the container."
    read -rp "Continue anyway? [y/N]: " CONT
    [[ "$CONT" =~ ^[Yy]$ ]] || err "Aborted by user."
fi

# ---------------------------------------------------------------------------
# Stop the container
# ---------------------------------------------------------------------------

info "Stopping container ${CONTAINER_ID} (if running)..."
pct stop "$CONTAINER_ID" >/dev/null 2>&1 || true

for _ in $(seq 1 30); do
    STATUS=$(pct status "$CONTAINER_ID" | awk '{print $2}')
    [ "$STATUS" = "stopped" ] && break
    sleep 1
done
[ "$(pct status "$CONTAINER_ID" | awk '{print $2}')" = "stopped" ] \
    || err "Container did not stop within 30s."

# ---------------------------------------------------------------------------
# Backup config (timestamped)
# ---------------------------------------------------------------------------

BACKUP_FILE="${CONFIG_FILE}.backup-$(date +%Y%m%d-%H%M%S)"
cp -a "$CONFIG_FILE" "$BACKUP_FILE"
info "Backup written to: ${BACKUP_FILE}"

# ---------------------------------------------------------------------------
# Append GPU passthrough block
# ---------------------------------------------------------------------------

info "Appending GPU passthrough configuration..."

{
    echo ""
    echo "$GPU_MARKER"
    echo "lxc.apparmor.profile: unconfined"
    echo "lxc.cap.drop:"
    echo "lxc.cgroup2.devices.allow: c ${CARD_MAJOR}:${CARD_MINOR} rwm"
    echo "lxc.cgroup2.devices.allow: c ${REND_MAJOR}:${REND_MINOR} rwm"
    echo "lxc.mount.entry: ${CARD_PATH} dev/dri/card0 none bind,optional,create=file"
    echo "lxc.mount.entry: ${RENDER_PATH} dev/dri/renderD128 none bind,optional,create=file"
    if [ "$HAS_KFD" -eq 1 ]; then
        echo "lxc.cgroup2.devices.allow: c ${KFD_MAJOR}:${KFD_MINOR} rwm"
        echo "lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file"
    fi
    echo "# --- end AMD GPU Passthrough ---"
} >> "$CONFIG_FILE"

# ---------------------------------------------------------------------------
# Start container
# ---------------------------------------------------------------------------

info "Starting container ${CONTAINER_ID}..."
if ! pct start "$CONTAINER_ID"; then
    warn "Container failed to start. Restoring previous config from ${BACKUP_FILE}."
    cp -a "$BACKUP_FILE" "$CONFIG_FILE"
    err "Aborted; original config restored."
fi

info "Done. Container ${CONTAINER_ID} upgraded with AMD GPU passthrough."
info "Inside the container, verify with:"
echo "    ls -l /dev/dri/"
[ "$HAS_KFD" -eq 1 ] && echo "    ls -l /dev/kfd"
echo "    rocminfo   # if ROCm is installed"
