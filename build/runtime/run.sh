#!/bin/sh

# The image ships one server, named by FALKORDB_SERVER (set in the Dockerfile):
# redis or valkey. Its arguments come from its own variable, REDIS_ARGS or
# VALKEY_ARGS, or from the other one when its own is unset, empty or only
# whitespace, so one setting works on either image. The two are never joined.
# Options only the other server understands (e.g. Redis-only configs in
# REDIS_ARGS on a valkey image) are passed through and make the server refuse
# to start: set the variable that matches the image.
SERVER="${FALKORDB_SERVER:-redis}"
case "$SERVER" in
    redis) own_args="${REDIS_ARGS:-}" other_args="${VALKEY_ARGS:-}" ;;
    valkey) own_args="${VALKEY_ARGS:-}" other_args="${REDIS_ARGS:-}" ;;
    *)
        echo "Error: FALKORDB_SERVER must be redis or valkey, not '$SERVER'." >&2
        exit 1
        ;;
esac
if ! command -v "${SERVER}-server" >/dev/null 2>&1; then
    echo "Error: ${SERVER}-server is not in this image; FALKORDB_SERVER is set by the image and should not be overridden." >&2
    exit 1
fi
case "$own_args" in
    *[![:space:]]*) SERVER_ARGS="$own_args" ;;
    *) SERVER_ARGS="$other_args" ;;
esac

# The server runs as the unprivileged `falkordb` user (uid 999, created in the
# Dockerfile), as in the official redis image. The container itself still
# starts as root: the browser entrypoint needs root to switch to nextjs, and a
# data volume written by an older image is owned by root and has to be handed
# over first. Root is kept when the container was started with --user, when
# SKIP_DROP_PRIVS is set, when it lacks CAP_SETUID/CAP_SETGID (e.g.
# --cap-drop ALL) and so could not switch users anyway, or when the files
# the server writes could not be handed to falkordb. SKIP_FIX_PERMS skips
# the ownership fix-up. Both names are the official image's.
FALKORDB_USER=falkordb

# has_cap BIT: is capability number BIT in this process's effective set?
has_cap() {
    while read -r key value; do
        if [ "$key" = "CapEff:" ]; then
            [ $(( (0x$value >> $1) & 1 )) -eq 1 ]
            return
        fi
    done < /proc/self/status
    return 1
}

# Hand the data dir to falkordb, but only when it holds nothing except redis's
# own files — the official image's guard, so a bind mount of anything else
# is never re-owned. The official list is *.rdb and appendonlydir; nodes*.conf
# is added because a cluster node's volume from a root-run image holds one,
# and as `falkordb` the node cannot rewrite it and exits at startup.
# Returns non-zero when the dir was not handed over.
fix_data_dir_perms() {
    unknown=$(find "${FALKORDB_DATA_PATH}" -mindepth 1 -maxdepth 1 \
        ! \( -name '*.rdb' -o -name 'nodes*.conf' -o \( -type d -name appendonlydir \) \) -print -quit)
    if [ -n "$unknown" ]; then
        echo "Notice: unknown file '$unknown' in ${FALKORDB_DATA_PATH}; ownership not changed."
        return 1
    fi
    if ! find "${FALKORDB_DATA_PATH}" ! -user "$FALKORDB_USER" -exec chown "$FALKORDB_USER:$FALKORDB_USER" {} +; then
        echo "Warning: could not hand ${FALKORDB_DATA_PATH} to $FALKORDB_USER."
        return 1
    fi
}

# gen-certs.sh runs as root and writes its keys 0600. Hand falkordb the files it
# generates, and only those, so a certificate mounted alongside is left alone.
# Returns non-zero when any of them was not handed over.
fix_tls_perms() {
    rc=0
    for f in ca.key ca.crt ca.txt openssl.cnf server.key server.crt \
             client.key client.crt redis.key redis.crt redis.dh; do
        if [ -e "${FALKORDB_TLS_PATH}/$f" ] &&
            ! chown "$FALKORDB_USER:$FALKORDB_USER" "${FALKORDB_TLS_PATH}/$f"; then
            echo "Warning: could not hand ${FALKORDB_TLS_PATH}/$f to $FALKORDB_USER."
            rc=1
        fi
    done
    return $rc
}

# Called when a fix-up above failed: falkordb could not use those files, so
# the server stays root, as it ran before this image dropped privileges.
keep_root() {
    echo "Notice: running ${SERVER}-server as root. Set SKIP_FIX_PERMS=1 to run it as $FALKORDB_USER regardless."
    drop_privs=""
}

# The Redis data check, on Valkey images only. Valkey cannot load the save
# files Redis 7.4 and later write: an RDB headed REDIS0 with version 12 or
# higher is refused unless rdb-version-check is relaxed (valkey rdb.c,
# rdbIsVersionAccepted). Redis 7.4 writes version 12, Redis 8.10 version 15.
# The server is always started with --dir "$FALKORDB_DATA_PATH", and it takes
# only plain file names for dbfilename, appendfilename and appenddirname, so it
# can load only the top-level files of that directory and the *.base.rdb in
# its AOF directory (the rest of an AOF is commands, which Valkey replays
# whichever server wrote them). Check the header of each, and stop with the
# migration steps when Redis wrote one. The migration is
# docs/valkey-migration.md.

# redis_rdb_version FILE: the RDB version when FILE starts with a Redis RDB
# header, REDIS0 and three digits (`REDIS0015`); nothing otherwise. Valkey 9
# heads its own files VALKEY.
redis_rdb_version() {
    _magic=$(head -c 9 "$1" 2>/dev/null) || return 0
    case "$_magic" in
        # 1NNN - 1000 rather than NNN: a leading 0 would read as octal.
        REDIS0[0-9][0-9][0-9]) echo $(( 1${_magic#REDIS0} - 1000 )) ;;
    esac
}

valkey_preflight() {
    # find, not a glob: a glob skips dot-prefixed names (--dbfilename ..x,
    # --appendfilename .aof, whose base file is .aof.1.base.rdb).
    redis_files=$(
        { find -L "${FALKORDB_DATA_PATH}" -maxdepth 1 -type f
          find -L "${FALKORDB_DATA_PATH}" -mindepth 2 -maxdepth 2 -type f -name '*.base.rdb'
        } 2>/dev/null | while IFS= read -r _f; do
            _version=$(redis_rdb_version "$_f")
            if [ -n "$_version" ] && [ "$_version" -ge 12 ]; then
                echo "  $_f (Redis RDB version $_version)"
            fi
        done
    )
    [ -z "$redis_files" ] && return 0
    cat >&2 <<EOF
Error: this image runs Valkey, and ${FALKORDB_DATA_PATH} holds save files
written by Redis 7.4 or later, which Valkey refuses to load:
${redis_files}
To keep this data, run a Redis image of FalkorDB (a tag ending in -redis) on
this volume, or migrate the data to Valkey by starting this image with
FALKORDB_REDIS_MIGRATION=1. Back up the volume first.
See https://github.com/FalkorDB/FalkorDB/blob/main/docs/valkey-migration.md
EOF
    exit 1
}

if [ "$SERVER" = "valkey" ]; then
    if [ "${FALKORDB_REDIS_MIGRATION:-0}" = "1" ]; then
        SERVER_ARGS="${SERVER_ARGS} --rdb-version-check relaxed"
        echo "Notice: FALKORDB_REDIS_MIGRATION=1: starting with --rdb-version-check relaxed. Save the data in Valkey's format (SAVE, and with AOF on, BGREWRITEAOF), then restart without it."
    else
        valkey_preflight
    fi
fi

if [ "${BROWSER:-1}" -eq "1" ]; then
    if [ -d "${FALKORDB_BROWSER_PATH}" ]; then
        cd "${FALKORDB_BROWSER_PATH}" && HOSTNAME="0.0.0.0" ./docker-entrypoint.sh node server.js &
    fi
fi

# Create /var/lib/falkordb/data directory if it does not exist.
# -p so the call is idempotent and creates parents if a bind-mount points
# at a deeper path that hasn't been pre-staged.
if [ ! -d "${FALKORDB_DATA_PATH}" ]; then
    mkdir -p "${FALKORDB_DATA_PATH}"
fi

drop_privs=""
if [ "$(id -u)" = "0" ] && [ -z "${SKIP_DROP_PRIVS:-}" ] && has_cap 6 && has_cap 7; then
    drop_privs=1
    if [ -z "${SKIP_FIX_PERMS:-}" ] && ! fix_data_dir_perms; then
        keep_root
    fi
fi

if [ "${TLS:-0}" -eq "1" ]; then
    # shellcheck disable=SC2086
    ${FALKORDB_BIN_PATH}/gen-certs.sh
    if [ -n "$drop_privs" ] && [ -z "${SKIP_FIX_PERMS:-}" ] && ! fix_tls_perms; then
        keep_root
    fi
fi

# Files the server writes (RDB, AOF) are private to it, as in the official image.
if [ "$(umask)" = "0022" ]; then
    umask 0077
fi

# The exec prefix: `gosu falkordb` when dropping root, nothing otherwise.
if [ -n "$drop_privs" ]; then
    set -- gosu "$FALKORDB_USER"
else
    set --
fi

if [ "${TLS:-0}" -eq "1" ]; then
    # shellcheck disable=SC2086
    exec "$@" "${SERVER}-server" ${SERVER_ARGS} --protected-mode no \
        --tls-port 6379 --port 0 \
        --tls-cert-file ${FALKORDB_TLS_PATH}/redis.crt \
        --tls-key-file ${FALKORDB_TLS_PATH}/redis.key \
        --tls-ca-cert-file ${FALKORDB_TLS_PATH}/ca.crt \
        --tls-auth-clients no \
        --dir "${FALKORDB_DATA_PATH}" \
        --loadmodule "${FALKORDB_BIN_PATH}/falkordb.so" ${FALKORDB_ARGS}
else
    # shellcheck disable=SC2086
    exec "$@" "${SERVER}-server" ${SERVER_ARGS} --protected-mode no \
        --dir "${FALKORDB_DATA_PATH}" \
        --loadmodule "${FALKORDB_BIN_PATH}/falkordb.so" ${FALKORDB_ARGS}
fi
