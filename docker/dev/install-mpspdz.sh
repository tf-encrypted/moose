#!/bin/bash

curl -L https://github.com/data61/MP-SPDZ/releases/download/v0.1.9/mp-spdz-0.1.9.tar.xz | tar xJv
mv mp-spdz-0.1.9 MP-SPDZ

cd MP-SPDZ
Scripts/tldr.sh

