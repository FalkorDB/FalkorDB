#!/bin/sh
# Builds the server FalkorDB runs on -- redis or valkey -- from its upstream
# release tag. Every image that ships a server (the toolchain, release, per-OS
# and coverage images) builds it through this one script, so the same server
# version comes out with the same flags everywhere. Docker has no include
# mechanism, so the Dockerfiles COPY this in rather than repeating the recipe.
# (The ASAN image keeps its own recipe: it builds a sanitizer server.)
#
# Usage: build-server.sh <redis|valkey> <version> <prefix>
#
# Installs exactly this under <prefix>, for `COPY --from=... <prefix>/ /usr/local/`
# (a directory COPY keeps the symlinks; a single-file COPY would dereference
# them and duplicate the server binary):
#   bin/<server>-server, bin/<server>-cli        stripped binaries
#   bin/<server>-{sentinel,check-rdb,check-aof}  symlinks to <server>-server,
#                                                which picks its mode from argv[0]
#   bin/redis-*                                  valkey only, unless
#                                                REDIS_ALIASES=no: symlinks to the
#                                                valkey-* names, which valkey
#                                                recognises in argv[0]
#   share/licenses/<server>/                     the upstream license
#
# Both servers get what the upstream docker images do: protected mode off by
# default (docker users supply auth/network controls externally), the same
# jemalloc page sizes on every build host, and TLS.
set -eu

if [ $# -ne 3 ]; then
    echo "usage: $0 <redis|valkey> <version> <prefix>" >&2
    exit 2
fi
server=$1 version=$2 prefix=$3

case "$server" in
    redis)
        repo=https://github.com/redis/redis.git
        license=LICENSE.txt
        # Since 8.10 the bundled modules (redisearch, rejson, ...) are cloned
        # on demand by `make modules-update`, so a plain build is redis-only.
        # BUILD_WITH_MODULES=no is the pre-8.10 spelling of the same intent,
        # kept so the recipe still builds redis-only on an 8.6.x tag.
        export BUILD_WITH_MODULES=no
        ;;
    valkey)
        repo=https://github.com/valkey-io/valkey.git
        license=COPYING
        # The redis-* aliases are made below, as symlinks into <prefix>.
        export USE_REDIS_SYMLINKS=no
        ;;
    *)
        echo "unknown server '$server' (expected redis or valkey)" >&2
        exit 2
        ;;
esac
export BUILD_TLS=yes

set -x
src="/tmp/${server}-src"
install_dir="/tmp/${server}-install"
# The tag ref is content-addressed by git, so no separate SHA pin is needed.
git clone --depth 1 --branch "$version" "$repo" "$src"
cd "$src"

sed -ri 's!^( *createBoolConfig[(]"protected-mode",.*, *)1( *,.*[)],)$!\10\2!' src/config.c
grep -E '^ *createBoolConfig[(]"protected-mode",.*, *0 *,.*[)],$' src/config.c

arch="$(uname -m)"
case "$arch" in
    x86_64 | i?86) lgPage=12 ;;
    *) lgPage=16 ;;
esac
jemallocFlags="--build=${arch}-linux-gnu --with-lg-page=${lgPage} --with-lg-hugepage=21"
sed -ri 's!cd jemalloc && ./configure !&'"$jemallocFlags"' !' deps/Makefile
grep -F "cd jemalloc && ./configure $jemallocFlags " deps/Makefile

make -j "$(nproc)" all
make PREFIX="$install_dir" install

mkdir -p "$prefix/bin" "$prefix/share/licenses/$server"
for tool in server cli; do
    install -m 0755 "$install_dir/bin/${server}-${tool}" "$prefix/bin/"
    strip "$prefix/bin/${server}-${tool}"
done
for mode in sentinel check-rdb check-aof; do
    ln -s "${server}-server" "$prefix/bin/${server}-${mode}"
done
if [ "$server" != redis ] && [ "${REDIS_ALIASES:-yes}" != no ]; then
    for tool in server cli sentinel check-rdb check-aof; do
        ln -s "${server}-${tool}" "$prefix/bin/redis-${tool}"
    done
fi
install -m 0644 "$license" "$prefix/share/licenses/$server/"

"$prefix/bin/${server}-server" --version | grep -F " v=${version} "
"$prefix/bin/${server}-cli" --version
