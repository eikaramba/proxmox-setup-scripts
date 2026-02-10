#!/usr/bin/env bash
# SCRIPT_DESC: Install AMD ROCm 7.2.X drivers
# SCRIPT_DETECT: lsmod | grep -q amdgpu

# if uninstall needed
# # Purge existing packages to resolve version conflicts (e.g. 7.2.0 vs 7.1.1)
# apt purge -y "rocm-*" "amdgpu-*" "hsakmt-*" "rock-dkms" "rocm-core"
# apt autoremove -y
# # Clean up potential leftover directories
# rm -rf /opt/rocm*
mkdir --parents /etc/apt/keyrings
chmod 0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.2/ubuntu noble main
EOF

tee /etc/apt/preferences.d/rocm-pin-600 << EOF
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

echo ">>> Updating package lists after adding ROCm repository"
apt update

echo ">>> Installing AMD ROCm drivers and tools"
apt install -y rocm-smi rocminfo rocm-libs
apt install -y radeontop

echo ">>> Adding root user to render and video groups for GPU access"
usermod -a -G render,video root

echo ">>> Verifying root user group membership"
groups root

echo ">>> Setting up environment variables for ROCm"
cat > /etc/profile.d/rocm.sh << 'EOF'
export PATH="${PATH:+${PATH}:}/opt/rocm/bin/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/opt/rocm/lib/"
EOF

echo ">>> Making /etc/profile.d/rocm.sh executable and sourcing it"
chmod +x /etc/profile.d/rocm.sh
# shellcheck disable=SC1091
source /etc/profile.d/rocm.sh

echo ">>> AMD ROCm driver installation completed."
echo ">>> Run '005 - verify-amd-drivers.sh' to verify the installation."
