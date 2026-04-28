#!/usr/bin/env bash
# SCRIPT_DESC: Create GPU-enabled LXC container (AMD or NVIDIA)
# SCRIPT_DETECT:

# Enhanced LXC GPU container creation script with automatic GPU detection.
# - Uses persistent PCI by-path device names for consistent mapping.
# - Derives cgroup major/minor from the SELECTED PCI device (not hardcoded card1).
# - Skips KFD config cleanly if /dev/kfd is absent.
# - Enables IPv6 SLAAC by default (toggleable).

set -e

# Get script directory and source colors
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../includes/colors.sh"

# ---------------------------------------------------------------------------
# Helper: extract "major minor" for a character device via stat (robust)
# ---------------------------------------------------------------------------
get_dev_major_minor() {
    local dev="$1"
    # stat -c '%t %T' returns hex; convert to decimal
    local hex_major hex_minor
    hex_major=$(stat -c '%t' "$dev")
    hex_minor=$(stat -c '%T' "$dev")
    printf '%d %d\n' "0x${hex_major}" "0x${hex_minor}"
}

# ---------------------------------------------------------------------------
# Prompt: container ID
# ---------------------------------------------------------------------------
read -r -p "Enter container ID [100]: " CONTAINER_ID
CONTAINER_ID=${CONTAINER_ID:-100}

# ---------------------------------------------------------------------------
# Prompt: GPU type
# ---------------------------------------------------------------------------
echo ""
echo "Select GPU type:"
echo "1) AMD GPU"
echo "2) NVIDIA GPU"
read -r -p "Enter selection [1]: " GPU_TYPE
GPU_TYPE=${GPU_TYPE:-1}
GPU_NAME=""
ADDITIONAL_TAGS=""

# ---------------------------------------------------------------------------
# Detect GPUs
# ---------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}>>> Detecting available GPUs...${NC}"
echo ""

TEMPLATE_FIRST_PCI_PATH=""

if [ "$GPU_TYPE" == "1" ]; then
    GPU_NAME="AMD"
    ADDITIONAL_TAGS="amd"
    echo "=== Available AMD GPUs ==="
    echo ""
    lspci -nn -D | grep -i amd | grep -i "VGA\|3D\|Display" && echo "" || echo "No AMD GPUs found via lspci"

    echo "Available AMD GPU PCI paths:"
    for card in /dev/dri/by-path/pci-*-card; do
        if [ -e "$card" ]; then
            pci_addr=$(basename "$card" | sed 's/pci-\(.*\)-card/\1/')
            gpu_info=$(lspci -s "${pci_addr#0000:}" 2>/dev/null | grep -i "VGA\|3D\|Display" || echo "")
            if echo "$gpu_info" | grep -qi amd; then
                echo "  $pci_addr -> $(ls -l "$card" | awk '{print $NF}') (AMD)"
                echo "    $gpu_info"
                if [ -z "$TEMPLATE_FIRST_PCI_PATH" ]; then
                    TEMPLATE_FIRST_PCI_PATH="$pci_addr"
                fi
            fi
        fi
    done
    echo ""
else
    GPU_NAME="NVIDIA"
    ADDITIONAL_TAGS="nvidia"
    echo "=== Available NVIDIA GPUs ==="
    echo ""
    lspci -nn -D | grep -i nvidia | grep -i "VGA\|3D\|Display" && echo "" || echo "No NVIDIA GPUs found"

    echo "Available NVIDIA GPU PCI paths:"
    for card in /dev/dri/by-path/pci-*-card; do
        if [ -e "$card" ]; then
            pci_addr=$(basename "$card" | sed 's/pci-\(.*\)-card/\1/')
            gpu_info=$(lspci -s "${pci_addr#0000:}" 2>/dev/null | grep -i "VGA\|3D\|Display" || echo "")
            if echo "$gpu_info" | grep -qi nvidia; then
                echo "  $pci_addr -> $(ls -l "$card" | awk '{print $NF}') (NVIDIA)"
                echo "    $gpu_info"
                if [ -z "$TEMPLATE_FIRST_PCI_PATH" ]; then
                    TEMPLATE_FIRST_PCI_PATH="$pci_addr"
                fi
            fi
        fi
    done
    echo ""
fi

# ---------------------------------------------------------------------------
# Prompt: PCI address with default
# ---------------------------------------------------------------------------
if [ -n "$TEMPLATE_FIRST_PCI_PATH" ]; then
    read -r -p "Enter GPU PCI address [$TEMPLATE_FIRST_PCI_PATH]: " PCI_ADDRESS
    PCI_ADDRESS=${PCI_ADDRESS:-$TEMPLATE_FIRST_PCI_PATH}
else
    read -r -p "Enter GPU PCI address (e.g., 0000:a1:00.0): " PCI_ADDRESS
fi

if [ -z "$PCI_ADDRESS" ]; then
    echo -e "${RED}Error: PCI address is required${NC}"
    exit 1
fi

# Validate paths
CARD_PATH="/dev/dri/by-path/pci-${PCI_ADDRESS}-card"
RENDER_PATH="/dev/dri/by-path/pci-${PCI_ADDRESS}-render"

if [ ! -e "$CARD_PATH" ]; then
    echo -e "${RED}Error: $CARD_PATH does not exist${NC}"
    exit 1
fi
if [ ! -e "$RENDER_PATH" ]; then
    echo -e "${RED}Error: $RENDER_PATH does not exist${NC}"
    exit 1
fi

# Resolve to real /dev/dri/cardN and /dev/dri/renderD12N (so cgroup major/minor are correct for the selected GPU)
REAL_CARD=$(readlink -f "$CARD_PATH")
REAL_RENDER=$(readlink -f "$RENDER_PATH")

if [ "$GPU_TYPE" == "1" ]; then
    if [ ! -e "/dev/kfd" ]; then
        echo -e "${YELLOW}Warning: /dev/kfd does not exist. AMD ROCm will be disabled in this container.${NC}"
        echo -e "${YELLOW}If you need ROCm, install AMD GPU drivers on the host first.${NC}"
    fi

    echo -e "${GREEN}✓ Found AMD GPU at $PCI_ADDRESS${NC}"
    echo "  Card device:   $CARD_PATH -> $REAL_CARD"
    echo "  Render device: $RENDER_PATH -> $REAL_RENDER"
    echo "  KFD device:    $([ -e "/dev/kfd" ] && echo "✓ Available" || echo "✗ Not found (ROCm disabled)")"
else
    echo -e "${GREEN}✓ Found NVIDIA GPU at $PCI_ADDRESS${NC}"
    echo "  Card device:   $CARD_PATH -> $REAL_CARD"
    echo "  Render device: $RENDER_PATH -> $REAL_RENDER"
    echo ""
    echo "Validating NVIDIA driver devices:"

    NVIDIA_DEVICES=("/dev/nvidia0" "/dev/nvidiactl" "/dev/nvidia-modeset" "/dev/nvidia-uvm")
    MISSING_DEVICES=()

    for dev in "${NVIDIA_DEVICES[@]}"; do
        if [ -e "$dev" ]; then
            echo "  ✓ $dev"
        else
            echo "  ✗ $dev (missing)"
            MISSING_DEVICES+=("$dev")
        fi
    done

    if [ ${#MISSING_DEVICES[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Warning: Some NVIDIA devices are missing:${NC}"
        for dev in "${MISSING_DEVICES[@]}"; do
            echo -e "${YELLOW}  - $dev${NC}"
        done
        echo -e "${YELLOW}Make sure NVIDIA drivers are properly installed on the host.${NC}"
        echo ""
        read -r -p "Continue anyway? [y/N]: " CONTINUE
        CONTINUE=${CONTINUE:-N}
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Prompt: hostname, network, storage
# ---------------------------------------------------------------------------
echo ""
HOSTNAME_TEMPLATE="ollama-docker-${GPU_NAME,,}-$CONTAINER_ID"
read -r -p "Enter hostname [$HOSTNAME_TEMPLATE]: " HOSTNAME
HOSTNAME=${HOSTNAME:-$HOSTNAME_TEMPLATE}

IP_TEMPLATE="10.0.0.$CONTAINER_ID"
read -r -p "Enter container IP address [$IP_TEMPLATE]: " IP_ADDRESS
IP_ADDRESS=${IP_ADDRESS:-$IP_TEMPLATE}

GW_TEMPLATE="10.0.0.1"
read -r -p "Enter gateway [$GW_TEMPLATE]: " GATEWAY
GATEWAY=${GATEWAY:-$GW_TEMPLATE}

# IPv6 toggle (defaults to enabled)
read -r -p "Enable IPv6 (SLAAC)? [Y/n]: " ENABLE_IPV6
ENABLE_IPV6=${ENABLE_IPV6:-Y}
if [[ "$ENABLE_IPV6" =~ ^[Yy]$ ]]; then
    IPV6_OPT=",ip6=auto"
    IPV6_DESC="Enabled (SLAAC)"
else
    IPV6_OPT=""
    IPV6_DESC="Disabled"
fi

ROOT_FS_LOCATION_TEMPLATE="local-lvm"
read -r -p "Enter root fs location [$ROOT_FS_LOCATION_TEMPLATE]: " ROOT_FS_LOCATION
ROOT_FS_LOCATION=${ROOT_FS_LOCATION:-$ROOT_FS_LOCATION_TEMPLATE}

# Random MAC
MAC_ADDRESS=$(printf 'BC:24:11:%02X:%02X:%02X\n' $((RANDOM%256)) $((RANDOM%256)) $((RANDOM%256)))

# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}>>> Configuration Summary${NC}"
echo "Container ID: $CONTAINER_ID"
echo "GPU Type:     $([ "$GPU_TYPE" == "1" ] && echo "AMD" || echo "NVIDIA")"
echo "PCI Address:  $PCI_ADDRESS"
echo "  -> card:    $REAL_CARD"
echo "  -> render:  $REAL_RENDER"
echo "Hostname:     $HOSTNAME"
echo "IP Address:   $IP_ADDRESS"
echo "Gateway:      $GATEWAY"
echo "IPv6:         $IPV6_DESC"
echo "MAC Address:  $MAC_ADDRESS"
echo "Root FS:      $ROOT_FS_LOCATION"
echo ""
read -r -p "Proceed with container creation? [Y/n]: " CONFIRM
CONFIRM=${CONFIRM:-Y}

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# ---------------------------------------------------------------------------
# Create container
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}>>> Updating Proxmox VE Appliance list${NC}"
pveam update

echo -e "${GREEN}>>> Downloading Ubuntu 24.04 LXC template to local storage${NC}"
pveam download local ubuntu-24.04-standard_24.04-2_amd64.tar.zst 2>/dev/null || echo "Template already exists"

echo -e "${GREEN}>>> Creating LXC container with GPU passthrough support${NC}"
pct create "$CONTAINER_ID" local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst \
    --arch amd64 \
    --cores 8 \
    --features nesting=1 \
    --hostname "$HOSTNAME" \
    --memory 8192 \
    --net0 "name=eth0,bridge=vmbr0,firewall=1,gw=$GATEWAY,hwaddr=$MAC_ADDRESS,ip=$IP_ADDRESS/24${IPV6_OPT},type=veth" \
    --ostype ubuntu \
    --password testing \
    --rootfs "$ROOT_FS_LOCATION:160" \
    --swap 4096 \
    --tags "docker;ollama;${ADDITIONAL_TAGS}" \
    --unprivileged 0

echo -e "${GREEN}>>> Added LXC container with ID $CONTAINER_ID${NC}"

# ---------------------------------------------------------------------------
# GPU passthrough configuration
# ---------------------------------------------------------------------------
if [ "$GPU_TYPE" == "1" ]; then
    echo -e "${GREEN}>>> Configuring AMD GPU passthrough${NC}"

    # cgroup major/minor for the SELECTED card and render nodes
    read -r CARD_MAJOR CARD_MINOR     < <(get_dev_major_minor "$REAL_CARD")
    read -r RENDER_MAJOR RENDER_MINOR < <(get_dev_major_minor "$REAL_RENDER")

    # KFD is optional — only emit lines if it exists
    if [ -e /dev/kfd ]; then
        read -r KFD_MAJOR KFD_MINOR < <(get_dev_major_minor "/dev/kfd")
        KFD_BLOCK="lxc.cgroup2.devices.allow: c ${KFD_MAJOR}:${KFD_MINOR} rwm
lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file"
    else
        KFD_BLOCK="# /dev/kfd not present on host; KFD/ROCm passthrough skipped"
    fi

    cat >> "/etc/pve/lxc/${CONTAINER_ID}.conf" << EOF
# ===== AMD GPU Passthrough Configuration =====
# PCI Address: $PCI_ADDRESS
# Card:   $REAL_CARD   (major:minor ${CARD_MAJOR}:${CARD_MINOR})
# Render: $REAL_RENDER (major:minor ${RENDER_MAJOR}:${RENDER_MINOR})
# Allow access to cgroup devices (DRI and KFD)
lxc.cgroup2.devices.allow: c ${CARD_MAJOR}:${CARD_MINOR} rwm
lxc.cgroup2.devices.allow: c ${RENDER_MAJOR}:${RENDER_MINOR} rwm
${KFD_BLOCK}
# Mount DRI devices using persistent PCI paths
lxc.mount.entry: /dev/dri/by-path/pci-${PCI_ADDRESS}-card dev/dri/card0 none bind,optional,create=file
lxc.mount.entry: /dev/dri/by-path/pci-${PCI_ADDRESS}-render dev/dri/renderD128 none bind,optional,create=file
# Allow system-level capabilities for GPU drivers
lxc.apparmor.profile: unconfined
lxc.cap.drop:
# ===== End GPU Configuration =====
EOF
else
    echo -e "${GREEN}>>> Configuring NVIDIA GPU passthrough${NC}"

    cat >> "/etc/pve/lxc/${CONTAINER_ID}.conf" << EOF
# ===== NVIDIA GPU Passthrough Configuration =====
# PCI Address: $PCI_ADDRESS
# Card:   $REAL_CARD
# Render: $REAL_RENDER
# Allow access to cgroup devices (NVIDIA and DRI)
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 226:* rwm
lxc.cgroup2.devices.allow: c 234:* rwm
lxc.cgroup2.devices.allow: c 237:* rwm
lxc.cgroup2.devices.allow: c 238:* rwm
lxc.cgroup2.devices.allow: c 239:* rwm
lxc.cgroup2.devices.allow: c 240:* rwm
lxc.cgroup2.devices.allow: c 508:* rwm
# Mount NVIDIA devices
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-modeset dev/nvidia-modeset none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-caps/nvidia-cap1 dev/nvidia-caps/nvidia-cap1 none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-caps/nvidia-cap2 dev/nvidia-caps/nvidia-cap2 none bind,optional,create=file
# Mount DRI devices using persistent PCI paths
lxc.mount.entry: /dev/dri/by-path/pci-${PCI_ADDRESS}-card dev/dri/card0 none bind,optional,create=file
lxc.mount.entry: /dev/dri/by-path/pci-${PCI_ADDRESS}-render dev/dri/renderD128 none bind,optional,create=file
# Allow system-level capabilities for GPU drivers
lxc.apparmor.profile: unconfined
lxc.cap.drop:
# ===== End GPU Configuration =====
EOF
fi

# ---------------------------------------------------------------------------
# Start, mount, configure
# ---------------------------------------------------------------------------
echo -e "${GREEN}>>> Starting container${NC}"
pct start "$CONTAINER_ID"
sleep 5

echo -e "${GREEN}>>> Mounting scripts directory into container${NC}"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
pct set "$CONTAINER_ID" -mp0 "$REPO_DIR,mp=/root/proxmox-setup-scripts"

echo -e "${GREEN}>>> Enabling SSH root login${NC}"
pct exec "$CONTAINER_ID" -- bash -c "sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config"
pct exec "$CONTAINER_ID" -- bash -c "sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config"
pct exec "$CONTAINER_ID" -- systemctl restart sshd

# ---------------------------------------------------------------------------
# Quick IPv6 sanity check (only if enabled)
# ---------------------------------------------------------------------------
if [ -n "$IPV6_OPT" ]; then
    echo ""
    echo -e "${GREEN}>>> Checking IPv6 connectivity in container...${NC}"
    sleep 3
    if pct exec "$CONTAINER_ID" -- ping6 -c 1 -W 2 2001:4860:4860::8888 >/dev/null 2>&1; then
        echo -e "${GREEN}✓ IPv6 OK${NC}"
    else
        echo -e "${YELLOW}⚠ IPv6 not yet reachable. This may resolve after a few seconds (SLAAC)."
        echo -e "  Verify later with: pct exec $CONTAINER_ID -- ping6 -c 3 google.com${NC}"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}>>> LXC Container Setup Complete! <<<${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Container ID:    $CONTAINER_ID"
echo "GPU Type:        $([ "$GPU_TYPE" == "1" ] && echo "AMD" || echo "NVIDIA")"
echo "GPU PCI Address: $PCI_ADDRESS"
echo "SSH Access:      ssh root@$IP_ADDRESS"
echo "IPv6:            $IPV6_DESC"
echo "Default Pwd:     testing"
echo "Scripts at:      /root/proxmox-setup-scripts"
echo ""
echo -e "${YELLOW}IMPORTANT: Change the default password after first login!${NC}"
echo ""
echo "To verify GPU inside container:"
if [ "$GPU_TYPE" == "1" ]; then
    echo "  pct exec $CONTAINER_ID -- ls -la /dev/dri/"
    echo "  pct exec $CONTAINER_ID -- ls -la /dev/kfd"
    echo ""
    read -r -p "Install Docker and AMD ROCm libraries now? [Y/n]: " RUN_INSTALL
    RUN_INSTALL=${RUN_INSTALL:-Y}

    if [[ "$RUN_INSTALL" =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${GREEN}>>> Running AMD GPU installation script...${NC}"
        pct exec "$CONTAINER_ID" -- bash /root/proxmox-setup-scripts/lxc/install-docker-and-amd-drivers-in-lxc.sh
    else
        echo ""
        echo -e "${YELLOW}Installation skipped. Run manually later:${NC}"
        echo "  pct exec $CONTAINER_ID -- bash /root/proxmox-setup-scripts/lxc/install-docker-and-amd-drivers-in-lxc.sh"
    fi
else
    echo "  pct exec $CONTAINER_ID -- ls -la /dev/nvidia*"
    echo "  pct exec $CONTAINER_ID -- ls -la /dev/dri/"
    echo ""
    read -r -p "Install Docker, NVIDIA libraries, and NVIDIA Container Toolkit now? [Y/n]: " RUN_INSTALL
    RUN_INSTALL=${RUN_INSTALL:-Y}

    if [[ "$RUN_INSTALL" =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${GREEN}>>> Running NVIDIA GPU installation script...${NC}"
        pct exec "$CONTAINER_ID" -- bash /root/proxmox-setup-scripts/lxc/install-docker-and-nvidia-drivers-in-lxc.sh
    else
        echo ""
        echo -e "${YELLOW}Installation skipped. Run manually later:${NC}"
        echo "  pct exec $CONTAINER_ID -- bash /root/proxmox-setup-scripts/lxc/install-docker-and-nvidia-drivers-in-lxc.sh"
    fi
fi
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}>>> Setup complete <<<${NC}"
echo -e "${GREEN}========================================${NC}"
