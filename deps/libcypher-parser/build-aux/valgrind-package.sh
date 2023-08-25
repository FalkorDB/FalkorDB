#!/bin/sh

set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 tarball distdir" >&2
    exit 1
fi

echo "Building..."
set -x
tar xzf $1
cd $2
./configure --enable-silent-rules --disable-maintainer-mode --disable-dependency-tracking --disable-debug
(cd lib/src && make all)
(cd lib/test && make check_libcypher-parser)
CK_FORK=no valgrind --leak-check=full --show-leak-kinds=all ./lib/test/check_libcypher-parser
