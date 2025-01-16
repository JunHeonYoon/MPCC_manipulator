#!/bin/sh
## Copyright 2019 Alexander Liniger

## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
###########################################################################
###########################################################################
## Install dependencies

set -e

# Default: do not install cv-plot
INSTALL_CVPLOT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-cvplot)
            INSTALL_CVPLOT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Repository links
repository_eigen="https://gitlab.com/libeigen/eigen.git"
repository_json="https://github.com/nlohmann/json.git"
repository_osqp="https://github.com/osqp/osqp"
repository_osqp_eigen="https://github.com/JunHeonYoon/osqp-eigen.git"
repository_rbdl="https://github.com/ORB-HD/rbdl-orb.git"
repository_cvplot="https://github.com/Profactor/cv-plot.git"

# Local installation folders
localFolder_eigen="External/Eigen"
localFolder_json="External/json"
localFolder_osqp="External/osqp"
localFolder_osqp_eigen="External/osqp_eigen"
localFolder_rbdl="External/rbdl-orb"
localFolder_cvplot="External/cv-plot"

if [ ! -d "$localFolder_eigen" ]; then
    echo "[INFO] Cloning Eigen..."
    git clone "$repository_eigen" "$localFolder_eigen"
else
    echo "[INFO] Eigen repository already cloned. Skipping."
fi

if [ ! -d "$localFolder_json" ]; then
    echo "[INFO] Cloning JSON..."
    git clone "$repository_json" "$localFolder_json"
else
    echo "[INFO] JSON repository already cloned. Skipping."
fi

if [ ! -d "$localFolder_osqp" ]; then
    echo "[INFO] Cloning OSQP..."
    git clone "$repository_osqp" "$localFolder_osqp"
else
    echo "[INFO] OSQP repository already cloned. Skipping."
fi

if [ ! -d "$localFolder_osqp_eigen" ]; then
    echo "[INFO] Cloning OSQP-Eigen..."
    git clone "$repository_osqp_eigen" "$localFolder_osqp_eigen"
else
    echo "[INFO] OSQP Eigen repository already cloned. Skipping."
fi

if [ ! -d "$localFolder_rbdl" ]; then
    echo "[INFO] Cloning RBDL..."
    git clone "$repository_rbdl" "$localFolder_rbdl"
else
    echo "[INFO] RBDL repository already cloned. Skipping."
fi

echo "[INFO] Installing osqp..."
cd $localFolder_osqp
mkdir -p build lib
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$(realpath ../lib)
make
make install
EXPORT_LINE="export LD_LIBRARY_PATH=\"$(realpath ../lib/lib)\":\$LD_LIBRARY_PATH"
if ! grep -Fxq "$EXPORT_LINE" ~/.bashrc
then
    echo "$EXPORT_LINE" >> ~/.bashrc
    echo "[INFO] Added LD_LIBRARY_PATH to ~/.bashrc"
else
    echo "[INFO] LD_LIBRARY_PATH already exists in ~/.bashrc. Skipping."
fi
. ~/.bashrc
cd ../../..

echo "[INFO] Installing rbdl..."
cd $localFolder_rbdl
mkdir -p build lib
cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$(realpath ../lib)
make
make install
cd ../../..


echo "[INFO] Installing osqp-eigen..."
cd $localFolder_osqp_eigen
mkdir -p build lib
cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$(realpath ../lib) -DCMAKE_PREFIX_PATH=$(realpath ../../../$localFolder_osqp/lib) -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make
make install
EXPORT_LINE="export OsqpEigen_DIR=\"$(realpath ..)\""
if ! grep -Fxq "$EXPORT_LINE" ~/.bashrc
then
    echo "$EXPORT_LINE" >> ~/.bashrc
    echo "[INFO] Added OsqpEigen_DIR to ~/.bashrc"
else
    echo "[INFO] OsqpEigen_DIR already exists in ~/.bashrc. Skipping."
fi
. ~/.bashrc
cd ../../..

echo "[INFO] Installing rbdl..."
cd $localFolder_rbdl
mkdir -p build lib
cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$(realpath ../lib)
make
make install
cd ../../..

if $INSTALL_CVPLOT; then
    if [ ! -d "$localFolder_cvplot" ]; then
        echo "[INFO] Cloning OSQP..."
        git clone "$repository_cvplot" "$localFolder_cvplot"
    else
        echo "[INFO] CV-Plot repository already cloned. Skipping."
    fi

    echo "[INFO] Installing cv-plot..."
    cd "$localFolder_cvplot"
    mkdir -p build lib
    cd build

    cmake .. -DCMAKE_INSTALL_PREFIX="$(realpath ../lib)" \
             -DCVPLOT_USE_CONAN=OFF \
             -DCVPLOT_HEADER_ONLY=OFF
    make
    make install
else
    echo "[INFO] Skipping cv-plot installation."
fi

