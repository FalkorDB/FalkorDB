#!/bin/sh
# Builds a pinned redis or valkey from its upstream release tag on macOS, for
# the macOS module build (.github/workflows/_build-macos.yml) to test against.
# The versions are the ones every Dockerfile pins, as build/check-server-pins.sh
# prints them, so a pin bump moves the macOS servers too.
#
# build/build-server.sh is the recipe for the images and stays Linux-only (GNU
# sed, nproc, jemalloc's linux-gnu page flags); changing it rebuilds every
# image. This is the macOS counterpart, deliberately plain: the system compiler,
# the server's own default allocator (libc malloc on macOS, as in any macOS
# build of these servers, including falkordblite's), no TLS, no bundled modules.
#
# Usage: build-server-macos.sh <redis|valkey> <version> <prefix>
#
# Installs <prefix>/bin/<server>-server and <prefix>/bin/<server>-cli.
set -eu

if [ $# -ne 3 ]; then
    echo "usage: $0 <redis|valkey> <version> <prefix>" >&2
    exit 2
fi
server=$1 version=$2 prefix=$3

case "$server" in
    redis)
        repo=https://github.com/redis/redis.git
        # Since 8.10 a plain build is redis-only; BUILD_WITH_MODULES=no is the
        # pre-8.10 spelling, kept as build-server.sh keeps it.
        export BUILD_WITH_MODULES=no
        ;;
    valkey)
        repo=https://github.com/valkey-io/valkey.git
        export USE_REDIS_SYMLINKS=no
        ;;
    *)
        echo "unknown server '$server' (expected redis or valkey)" >&2
        exit 2
        ;;
esac

# The servers are built with the system toolchain, whatever CC/CXX the caller
# exported for the module build.
unset CC CXX CFLAGS CXXFLAGS LDFLAGS

src=$(mktemp -d "${TMPDIR:-/tmp}/${server}-src.XXXXXX")
trap 'rm -rf "$src"' EXIT
# By tag, as build/build-server.sh does: upstream release tags are not moved.
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$version" "$repo" "$src"
make -C "$src" -j "$(sysctl -n hw.ncpu)" all >"$src/build.log" 2>&1 || {
    tail -50 "$src/build.log" >&2
    exit 1
}

mkdir -p "$prefix/bin"
for tool in server cli; do
    install -m 0755 "$src/src/${server}-${tool}" "$prefix/bin/"
    # -x: local symbols only. valkey looks its static Lua module up in its own
    # binary (ValkeyModule_OnLoad_lua), which a plain macOS strip removes.
    strip -x "$prefix/bin/${server}-${tool}"
done

"$prefix/bin/${server}-server" --version | grep -F " v=${version} "
"$prefix/bin/${server}-cli" --version
