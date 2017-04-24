#!/bin/bash

# Download the precompiled libraries of ilupack v2.4 for GNU64
# It must be executed in this directory

curl -L http://www.icm.tu-bs.de/~bolle/ilupack/download/ilupack05102016.zip | \
        bsdtar zxf - --strip-components 1 ilupack/lib/GNU64 &&
        rm -f lib/GNU64/libilupack_mc64.a
