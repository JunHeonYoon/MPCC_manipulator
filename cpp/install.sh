#!/bin/sh
## Copyright 2019 Alexander Liniger

## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at

##     http://www.apache.org/licenses/LICENSE-2.0

## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
###########################################################################
###########################################################################
## Install dependencies
set -e

## clone cv-plot
repository_cvplot="https://github.com/Profactor/cv-plot.git"
localFolder_cvplot="External/cv-plot"
git clone "$repository_cvplot" "$localFolder_cvplot"


cd External/cv-plot
mkdir -p build
mkdir -p lib
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$(realpath ../lib) -DCVPLOT_USE_CONAN=OFF -DCVPLOT_HEADER_ONLY=OFF
make
make install