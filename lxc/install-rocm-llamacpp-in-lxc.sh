#!/usr/bin/env bash

# ROCm + llama.cpp installation for LXC containers (NO Docker)
# This script installs AMD ROCm libraries and compiles llama.cpp with HIP support
# Designed for AMD Strix Halo (gfx1151) on Ubuntu 24.04 LXC based on https://github.com/kyuz0/amd-strix-halo-toolboxes/blob/main/toolboxes/Dockerfile.rocm-7.1.1-rocwmma

set -e

# Get script directory and source colors
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../includes/colors.sh"

# Configuration
ROCM_VERSION="7.1.1"
GPU_TARGET="gfx1151"  # AMD Strix Halo
LLAMA_CPP_DIR="/opt/llama.cpp"
ROCWMMA_ENABLED=true  # Enable rocWMMA for improved performance

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}ROCm + llama.cpp Setup for LXC${NC}"
echo -e "${GREEN}(No Docker - Native Installation)${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo -e "${YELLOW}Target GPU: AMD Strix Halo (${GPU_TARGET})${NC}"
echo -e "${YELLOW}ROCm Version: ${ROCM_VERSION}${NC}"
echo -e "${YELLOW}rocWMMA: $([ "$ROCWMMA_ENABLED" = true ] && echo "Enabled" || echo "Disabled")${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT: Make sure AMD drivers are installed on the Proxmox HOST first!${NC}"
echo -e "${YELLOW}Run '003 - install-amd-drivers.sh' on the host if not already done.${NC}"
echo ""

# Verify GPU is visible
echo -e "${GREEN}>>> Checking if GPU devices are accessible...${NC}"
GPU_FOUND=false
if [ -e /dev/kfd ]; then
    echo -e "${GREEN}✓ AMD GPU devices found:${NC}"
    ls -la /dev/kfd 2>/dev/null || true
    GPU_FOUND=true
fi

if [ -e /dev/dri/card0 ]; then
    echo -e "${GREEN}✓ DRI devices found:${NC}"
    ls -la /dev/dri/ 2>/dev/null || true
    GPU_FOUND=true
fi

if [ "$GPU_FOUND" = false ]; then
    echo -e "${RED}WARNING: No GPU devices found!${NC}"
    echo -e "${YELLOW}Make sure the LXC container has GPU passthrough configured correctly.${NC}"
    echo ""
    read -r -p "Continue anyway? [y/N]: " CONTINUE
    CONTINUE=${CONTINUE:-N}
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Cancelled.${NC}"
        exit 1
    fi
fi
echo ""

# Update package list and upgrade existing packages
echo -e "${GREEN}>>> Updating system packages...${NC}"
apt update && apt upgrade -y


# Install build prerequisites (including clang for OpenMP/ROCm)
echo -e "${GREEN}>>> Installing build prerequisites...${NC}"
apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    sudo \
    pciutils \
    ca-certificates \
    gnupg \
    lsb-release \
    pkg-config \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    clang

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Installing AMD ROCm ${ROCM_VERSION}${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

# Add AMD ROCm repository
echo -e "${GREEN}>>> Adding AMD ROCm repository...${NC}"
sudo mkdir --parents --mode=0755 /etc/apt/keyrings

# Download the ROCm GPG key
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Add ROCm repository (for Ubuntu 24.04 Noble)
sudo tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/${ROCM_VERSION}/ubuntu noble main
EOF

sudo tee /etc/apt/preferences.d/rocm-pin-600 << EOF
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

apt update

# Install ROCm development packages (needed for building llama.cpp)
echo -e "${GREEN}>>> Installing ROCm development packages...${NC}"
apt install -y \
    rocm-llvm \
    rocm-device-libs \
    hip-runtime-amd \
    hip-dev \
    rocblas \
    rocblas-dev \
    hipblas \
    hipblas-dev \
    rocm-cmake \
    rocm-core \
    rocm-dev \
    rocminfo \
    rocm-smi-lib \
    libomp-dev \
    libomp5

# Install monitoring tools
echo -e "${GREEN}>>> Installing monitoring tools...${NC}"
apt install -y nvtop radeontop || true

# Add root user to render and video groups (critical for GPU access)
usermod -a -G render,video root

# Set up ROCm environment variables
echo -e "${GREEN}>>> Setting up ROCm environment...${NC}"
cat > /etc/profile.d/rocm.sh << 'EOF'
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HIP_CLANG_PATH=/opt/rocm/llvm/bin
export HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export ROCBLAS_USE_HIPBLASLT=1
EOF

chmod +x /etc/profile.d/rocm.sh

# Also add to root's bashrc
cat >> /root/.bashrc << 'EOF'

# ROCm Environment Variables
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HIP_CLANG_PATH=/opt/rocm/llvm/bin
export HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export ROCBLAS_USE_HIPBLASLT=1
EOF

# Source the environment
source /etc/profile.d/rocm.sh

# Verify ROCm installation
echo ""
echo -e "${GREEN}>>> Verifying ROCm installation...${NC}"
which rocm-smi rocminfo || true
rocminfo | grep -i -A5 'Agent [0-9]' || echo -e "${YELLOW}rocminfo check skipped (may need reboot)${NC}"

if [ "$ROCWMMA_ENABLED" = true ]; then
    echo ""
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}Building rocWMMA${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    
    cd /opt
    rm -rf rocWMMA
    git clone https://github.com/ROCm/rocWMMA
    cd rocWMMA
    
    # Find libomp
    CANDIDATES=(
        "${ROCM_PATH}/llvm/lib/libomp.so"
        "${ROCM_PATH}/llvm/lib/libomp.a"
        "/usr/lib/x86_64-linux-gnu/libomp.so"
        "/usr/lib64/libomp.so"
        "/usr/lib64/libomp.a"
        "/usr/local/lib/libomp.so"
    )
    FOUND_LIBOMP=""
    for p in "${CANDIDATES[@]}"; do
        if [ -f "$p" ]; then
            FOUND_LIBOMP="$p"
            break
        fi
    done
    
    CMAKE_OPTS=""
    if [ -n "$FOUND_LIBOMP" ]; then
        OMP_LIB_DIR="$(dirname "$FOUND_LIBOMP")"
        CMAKE_OPTS="${CMAKE_OPTS} -DOpenMP_CXX_FLAGS=-fopenmp=libomp"
        CMAKE_OPTS="${CMAKE_OPTS} -DOpenMP_C_FLAGS=-fopenmp=libomp"
        CMAKE_OPTS="${CMAKE_OPTS} -DOpenMP_CXX_LIB_NAMES=omp"
        CMAKE_OPTS="${CMAKE_OPTS} -DOpenMP_C_LIB_NAMES=omp"
        CMAKE_OPTS="${CMAKE_OPTS} -DOpenMP_LIBRARY=${FOUND_LIBOMP}"
        CMAKE_OPTS="${CMAKE_OPTS} -DOpenMP_INCLUDE_DIR=${ROCM_PATH}/llvm/include"
        export LD_LIBRARY_PATH="${OMP_LIB_DIR}${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
        export CXXFLAGS="-fopenmp=libomp ${CXXFLAGS:-}"
        export LDFLAGS="-L${OMP_LIB_DIR} -lomp ${LDFLAGS:-}"
    else
        CMAKE_OPTS="${CMAKE_OPTS} -DOpenMP_CXX_FLAGS=-fopenmp=libomp -DOpenMP_C_FLAGS=-fopenmp=libomp"
        export CXXFLAGS="-fopenmp=libomp ${CXXFLAGS:-}"
        export LDFLAGS="${LDFLAGS:-} -lomp"
    fi
    
    # Find OpenMP include directory
    OMP_INCLUDE_DIR=""
    for dir in "/usr/lib/llvm-*/lib/clang/*/include" "/usr/lib64/clang/*/include" "${ROCM_PATH}/llvm/lib/clang/*/include"; do
        found_dir=$(ls -d $dir 2>/dev/null | head -1)
        if [ -n "$found_dir" ] && [ -d "$found_dir" ]; then
            OMP_INCLUDE_DIR="$found_dir"
            break
        fi
    done
    
    echo -e "${GREEN}>>> Building rocWMMA with Ninja...${NC}"
    CC=$ROCM_PATH/llvm/bin/amdclang \
    CXX=$ROCM_PATH/llvm/bin/amdclang++ \
    cmake -B build -S . -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$ROCM_PATH \
        -DROCWMMA_BUILD_TESTS=OFF \
        -DROCWMMA_BUILD_SAMPLES=OFF \
        -DGPU_TARGETS="${GPU_TARGET}" \
        -DOpenMP_CXX_FLAGS="-fopenmp=libomp" \
        -DOpenMP_C_FLAGS="-fopenmp=libomp" \
        ${FOUND_LIBOMP:+-DOpenMP_omp_LIBRARY="${FOUND_LIBOMP}"} \
        -DOpenMP_CXX_LIB_NAMES="omp" \
        -DOpenMP_C_LIB_NAMES="omp" \
        ${OMP_INCLUDE_DIR:+-DOpenMP_INCLUDE_DIRS="${OMP_INCLUDE_DIR}"}
    
    cmake --build build
    sudo cmake --install build
    
    echo -e "${GREEN}✓ rocWMMA installed successfully${NC}"
fi

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Cloning and Building llama.cpp${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

# Clone llama.cpp
echo -e "${GREEN}>>> Cloning llama.cpp...${NC}"
rm -rf "${LLAMA_CPP_DIR}"
mkdir -p "${LLAMA_CPP_DIR}"
cd "${LLAMA_CPP_DIR}"
git clone --recursive https://github.com/ggerganov/llama.cpp .

if [ "$ROCWMMA_ENABLED" = true ]; then
    echo ""
    echo -e "${GREEN}>>> Applying rocWMMA compatibility fixes...${NC}"
    
    # Apply rocWMMA fix inline
    VENDOR_HIP_FILE="${LLAMA_CPP_DIR}/ggml/src/ggml-cuda/vendors/hip.h"
    
    if [ -f "$VENDOR_HIP_FILE" ]; then
        # Check if fixes are already applied
        if grep -q "GGML_HIP_WARP_MASK" "$VENDOR_HIP_FILE" 2>/dev/null; then
            echo "rocWMMA fixes appear to already be applied (found GGML_HIP_WARP_MASK)"
        else
            echo "Step 1: Modifying HIP vendor header..."
            
            # Backup the original file
            cp "$VENDOR_HIP_FILE" "$VENDOR_HIP_FILE.backup"
            
            # Find the line with __shfl_sync definition
            SHFL_LINE=$(grep -n "^#define __shfl_sync" "$VENDOR_HIP_FILE" | head -1 | cut -d: -f1)
            
            if [ -n "$SHFL_LINE" ]; then
                # Create a temporary file with the fix
                {
                    head -n $((SHFL_LINE - 1)) "$VENDOR_HIP_FILE"
                    
                    cat << 'ROCWMMA_FIX'
#ifdef GGML_HIP_ROCWMMA_FATTN
// ROCm requires 64-bit masks for __shfl_*_sync functions
#define GGML_HIP_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#else
#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define GGML_HIP_WARP_MASK 0xFFFFFFFF
#endif
ROCWMMA_FIX
                    
                    tail -n +$((SHFL_LINE + 2)) "$VENDOR_HIP_FILE"
                } > "$VENDOR_HIP_FILE.tmp"
                
                mv "$VENDOR_HIP_FILE.tmp" "$VENDOR_HIP_FILE"
                echo "  ✓ Added conditional GGML_HIP_WARP_MASK macro to vendor header"
                
                echo ""
                echo "Step 2: Replacing hardcoded warp masks in CUDA files..."
                
                CUDA_FILES=$(find "${LLAMA_CPP_DIR}/ggml/src/ggml-cuda" -name "*.cu" -o -name "*.cuh" 2>/dev/null | sort)
                MODIFIED_COUNT=0
                
                for file in $CUDA_FILES; do
                    if grep -q "0xFFFFFFFF\|0xffffffff" "$file" 2>/dev/null; then
                        cp "$file" "$file.backup"
                        sed -i 's/0xFFFFFFFF/GGML_HIP_WARP_MASK/g; s/0xffffffff/GGML_HIP_WARP_MASK/g' "$file"
                        MODIFIED_COUNT=$((MODIFIED_COUNT + 1))
                        echo "  ✓ Modified: $(basename "$file")"
                    fi
                done
                
                echo "  ✓ Modified $MODIFIED_COUNT CUDA files"
                echo ""
                echo -e "${GREEN}✓ rocWMMA compatibility fixes applied successfully${NC}"
            else
                echo -e "${YELLOW}Warning: Could not find __shfl_sync macro definition${NC}"
                echo -e "${YELLOW}Continuing without rocWMMA fixes...${NC}"
                ROCWMMA_ENABLED=false
            fi
        fi
    else
        echo -e "${YELLOW}Warning: HIP vendor header not found at: $VENDOR_HIP_FILE${NC}"
        echo -e "${YELLOW}Continuing without rocWMMA fixes...${NC}"
        ROCWMMA_ENABLED=false
    fi
fi


# Build llama.cpp
echo ""
echo -e "${GREEN}>>> Building llama.cpp with HIP support...${NC}"
cd "${LLAMA_CPP_DIR}"
git clean -xdf
git submodule update --recursive

# Use clang/clang++ for OpenMP/ROCm compatibility
export CC=clang
export CXX=clang++

# Clean previous build directory to avoid cached compiler settings
rm -rf build

# Set cmake options
CMAKE_OPTS=(
    -DGGML_HIP=ON
    -DAMDGPU_TARGETS="${GPU_TARGET}"
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_RPC=ON
    -DLLAMA_HIP_UMA=ON
    -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON
    -DROCM_PATH=/opt/rocm
    -DHIP_PATH=/opt/rocm
    -DHIP_PLATFORM=amd
    -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm"
)

if [ "$ROCWMMA_ENABLED" = true ]; then
    CMAKE_OPTS+=(-DGGML_HIP_ROCWMMA_FATTN=ON)
fi

echo -e "${GREEN}>>> Running cmake configure...${NC}"
cmake -S . -B build "${CMAKE_OPTS[@]}"

echo -e "${GREEN}>>> Compiling llama.cpp (this may take a while)...${NC}"
cmake --build build --config Release -- -j$(nproc)

echo -e "${GREEN}>>> Installing llama.cpp...${NC}"
sudo cmake --install build --config Release

# Copy libraries to system lib directory
echo -e "${GREEN}>>> Setting up shared libraries...${NC}"
find "${LLAMA_CPP_DIR}/build" -type f -name 'lib*.so*' -exec cp {} /usr/local/lib/ \;

# Set up library paths
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/llama-cpp.conf
sudo ldconfig

# Create symlinks to binaries in /usr/local/bin (if not already there)
if [ -d "${LLAMA_CPP_DIR}/build/bin" ]; then
    for bin in "${LLAMA_CPP_DIR}/build/bin"/*; do
        if [ -f "$bin" ] && [ -x "$bin" ]; then
            binname=$(basename "$bin")
            if [ ! -f "/usr/local/bin/$binname" ]; then
                sudo cp "$bin" "/usr/local/bin/$binname"
            fi
        fi
    done
fi

# Create GGUF VRAM estimator utility
echo -e "${GREEN}>>> Installing GGUF VRAM estimator utility...${NC}"
cat > /usr/local/bin/gguf-vram-estimator.py << 'VRAM_ESTIMATOR'
#!/usr/bin/env python3
import sys
import os
import re
import struct
import argparse
from typing import Dict, Any, List

# GGUF constants
GGUF_MAGIC = 0x46554747
GGUF_VALUE_TYPE = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32",
    5: "INT32", 6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY",
}

class GGUFMetadataReader:
    """A minimal reader to get only the necessary KV metadata for cache calculation."""
    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, Any] = {}

    def read(self):
        with open(self.path, "rb") as f:
            self.f = f
            magic, _, _, metadata_kv_count = struct.unpack("<IIQQ", self.f.read(24))
            if magic != GGUF_MAGIC:
                raise ValueError("Invalid GGUF magic number")
            self._read_metadata(metadata_kv_count)
        return self

    def _read_string(self) -> str:
        (length,) = struct.unpack("<Q", self.f.read(8))
        return self.f.read(length).decode("utf-8", errors="replace")

    def _read_value(self, value_type_idx: int):
        value_type = GGUF_VALUE_TYPE.get(value_type_idx)
        if not value_type:
            raise ValueError(f"Unknown GGUF value type: {value_type_idx}")
        if value_type == "STRING":
            return self._read_string()
        if value_type == "UINT32":
            return struct.unpack("<I", self.f.read(4))[0]
        if value_type == "INT32":
            return struct.unpack("<i", self.f.read(4))[0]
        self._skip_value(value_type_idx)

    def _skip_value(self, value_type_idx: int):
        value_type = GGUF_VALUE_TYPE.get(value_type_idx)
        if not value_type:
            return
        if value_type in ("UINT8", "INT8", "BOOL"):
            self.f.seek(1, 1)
        elif value_type in ("UINT16", "INT16"):
            self.f.seek(2, 1)
        elif value_type in ("UINT32", "INT32", "FLOAT32"):
            self.f.seek(4, 1)
        elif value_type == "STRING":
            (length,) = struct.unpack("<Q", self.f.read(8))
            self.f.seek(length, 1)
        elif value_type == "ARRAY":
            (array_type_idx, count) = struct.unpack("<IQ", self.f.read(12))
            type_map = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
            element_size = type_map.get(array_type_idx)
            if element_size:
                self.f.seek(count * element_size, 1)
            else:
                for _ in range(count):
                    self._skip_value(8)

    def _read_metadata(self, count: int):
        keys_to_read = {"general.architecture", "general.name"}
        arch_specific_keys_added = False
        for _ in range(count):
            key = self._read_string()
            (value_type_idx,) = struct.unpack("<I", self.f.read(4))
            if not arch_specific_keys_added and "general.architecture" in self.metadata:
                prefix = self.metadata["general.architecture"]
                keys_to_read.update({
                    f"{prefix}.block_count", f"{prefix}.context_length",
                    f"{prefix}.attention.head_count_kv", f"{prefix}.attention.key_length",
                    f"{prefix}.attention.value_length", f"{prefix}.attention.sliding_window_size"
                })
                arch_specific_keys_added = True
            if key in keys_to_read:
                self.metadata[key] = self._read_value(value_type_idx)
            else:
                self._skip_value(value_type_idx)

def get_total_model_size_from_disk(gguf_file_path: str) -> int:
    """Calculates the total model size by finding all parts on disk."""
    match = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', gguf_file_path, re.IGNORECASE)
    if not match:
        return os.path.getsize(gguf_file_path)

    base_path = gguf_file_path[:match.start()]
    total_parts_str = match.group(2)
    total_parts = int(total_parts_str)
    total_size, found_parts = 0, 0
    for i in range(1, total_parts + 1):
        part_file_name = f"{base_path}-{i:05d}-of-{total_parts_str}.gguf"
        if os.path.exists(part_file_name):
            total_size += os.path.getsize(part_file_name)
            found_parts += 1
    if found_parts != total_parts:
        print(f"WARNING: Expected {total_parts} parts, found {found_parts}. Size calculation may be incomplete.", file=sys.stderr)
    return total_size

def format_mem(size_bytes):
    mib = size_bytes / (1024 * 1024)
    if mib < 1024:
        return f"{mib:8.2f} MiB"
    return f"{mib / 1024:8.2f} GiB"

def run_estimator(gguf_file: str, context_sizes: List[int], overhead_gib: float):
    try:
        reader = GGUFMetadataReader(gguf_file).read()
        metadata = reader.metadata
        prefix = metadata.get("general.architecture")
        if not prefix:
            raise KeyError("Could not read 'general.architecture' from model metadata.")

        model_size_bytes = get_total_model_size_from_disk(gguf_file)
        overhead_bytes = int(overhead_gib * 1024**3)

        n_layers = metadata[f"{prefix}.block_count"]
        n_head_kv = metadata[f"{prefix}.attention.head_count_kv"]
        training_context = metadata.get(f"{prefix}.context_length", 0)
        n_embd_head_k = metadata[f"{prefix}.attention.key_length"]
        n_embd_head_v = metadata[f"{prefix}.attention.value_length"]
        swa_window_size = metadata.get(f"{prefix}.attention.sliding_window_size", 0)

        is_scout_model = "scout" in metadata.get("general.name", "").lower()
        if is_scout_model and swa_window_size == 0:
            n_layers_swa, n_layers_full, swa_window_size = 36, 12, 8192
        elif swa_window_size > 0:
            n_layers_swa, n_layers_full = n_layers, 0
        else:
            n_layers_swa, n_layers_full = 0, n_layers

        print(f"\n--- Model '{metadata.get('general.name', 'N/A')}' ---")
        if training_context > 0:
            print(f"Max Context: {training_context:,} tokens")
        print(f"Model Size: {format_mem(model_size_bytes).strip()} (from file size)")
        print(f"Incl. Overhead: {overhead_gib:.2f} GiB (for compute buffer, etc. adjustable via --overhead)")

        if training_context > 0:
            context_sizes = sorted(list(set([c for c in context_sizes if c <= training_context] + [c for c in [training_context] if c not in context_sizes])))
        else:
            context_sizes = sorted(context_sizes)

        bytes_per_token_per_layer = n_head_kv * (n_embd_head_k + n_embd_head_v) * 2

        print("\n--- Memory Footprint Estimation ---")
        print(f"{'Context Size':>15s} | {'Context Memory':>15s} | {'Est. Total VRAM':>15s}")
        print("-" * 51)
        for n_ctx in context_sizes:
            mem_full = n_ctx * n_layers_full * bytes_per_token_per_layer
            mem_swa = min(n_ctx, swa_window_size) * n_layers_swa * bytes_per_token_per_layer
            kv_cache_bytes = mem_full + mem_swa
            total_bytes = model_size_bytes + kv_cache_bytes + overhead_bytes
            print(f"{n_ctx:>15,} | {format_mem(kv_cache_bytes):>15s} | {format_mem(total_bytes):>15s}")

    except (FileNotFoundError, ValueError, struct.error, NotImplementedError, KeyError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate VRAM requirements for a GGUF model, including a configurable overhead for compute buffers.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("gguf_file", help="Path to the GGUF model file (any part of a multi-part model).")
    parser.add_argument("-c", "--contexts", nargs='+', type=int, default=[4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576], help="Space-separated list of context sizes to calculate.")
    parser.add_argument("--overhead", type=float, default=2.0, help="Estimated overhead in GiB for compute buffers, drivers, etc. (default: 2.0)")
    args = parser.parse_args()
    run_estimator(args.gguf_file, args.contexts, args.overhead)

if __name__ == "__main__":
    main()
VRAM_ESTIMATOR

chmod +x /usr/local/bin/gguf-vram-estimator.py

# Verify installation
echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Verifying Installation${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

echo -e "${GREEN}>>> Checking installed binaries...${NC}"
which llama-cli llama-server rpc-server 2>/dev/null || echo -e "${YELLOW}Binaries may be in ${LLAMA_CPP_DIR}/build/bin${NC}"
ls -la /usr/local/bin/llama* /usr/local/bin/rpc* 2>/dev/null || true

echo ""
echo -e "${GREEN}>>> Testing ROCm...${NC}"
rocm-smi --showmemuse --showuse --showmeminfo all --showhw --showproductname 2>/dev/null || echo -e "${YELLOW}ROCm test skipped (may need reboot)${NC}"

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo -e "${GREEN}Installed components:${NC}"
echo -e "  • ROCm ${ROCM_VERSION}"
[ "$ROCWMMA_ENABLED" = true ] && echo -e "  • rocWMMA (for improved performance)"
echo -e "  • llama.cpp with HIP support (target: ${GPU_TARGET})"
echo -e "  • GGUF VRAM estimator utility"
echo ""
echo -e "${GREEN}Binary locations:${NC}"
echo -e "  • llama-cli, llama-server: /usr/local/bin/"
echo -e "  • rpc-server: /usr/local/bin/"
echo -e "  • Source: ${LLAMA_CPP_DIR}"
echo ""
echo -e "${GREEN}Quick start examples:${NC}"
echo -e "  # Run inference"
echo -e "  llama-cli -m /path/to/model.gguf -p \"Hello world\" -n 128"
echo ""
echo -e "  # Start API server"
echo -e "  llama-server -hf unsloth/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_XL -ngl 99 --threads -1 --ctx-size 32684 --port 8080 --host 0.0.0.0
echo ""
echo -e "  # Start RPC server for distributed inference"
echo -e "  rpc-server --host 0.0.0.0 --port 50052"
echo ""
echo -e "  # Estimate VRAM usage"
echo -e "  gguf-vram-estimator.py /path/to/model.gguf"
echo ""
echo -e "${YELLOW}NOTE: You may need to log out and back in (or reboot) for environment${NC}"
echo -e "${YELLOW}variables to take effect in new shells.${NC}"
echo ""
