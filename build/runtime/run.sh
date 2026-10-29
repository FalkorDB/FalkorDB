#!/bin/sh

# redis-server runs as the unprivileged `falkordb` user (uid 999, created in the
# Dockerfile), as in the official redis image. The container itself still
# starts as root: the browser entrypoint needs root to switch to nextjs, and a
# data volume written by an older image is owned by root and has to be handed
# over first. Root is kept when the container was started with --user, when
# SKIP_DROP_PRIVS is set, when it lacks CAP_SETUID/CAP_SETGID (e.g.
# --cap-drop ALL) and so could not switch users anyway, or when the files
# redis-server writes could not be handed to falkordb. SKIP_FIX_PERMS skips
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
    if ! find "${FALKORDB_DATA_PATH}" ! -user "$FALKORDB_USER" -exec chown -h "$FALKORDB_USER:$FALKORDB_USER" {} +; then
        echo "Warning: could not hand ${FALKORDB_DATA_PATH} to $FALKORDB_USER."
        return 1
    fi
}

# The files gen-certs.sh generates in the TLS directory.
TLS_FILES="ca.key ca.crt ca.txt openssl.cnf server.key server.crt
           client.key client.crt redis.key redis.crt redis.dh"
# A CA supplied to sign with: gen-certs.sh only reads these when they exist.
TLS_CA_FILES="ca.key ca.crt"

is_tls_ca_file() {
    case " $TLS_CA_FILES " in
        *" $1 "*) return 0 ;;
    esac
    return 1
}

# Before root chowns those files, or generates them where falkordb cannot
# write, each name must be absent or a regular file with a single link: a
# symlink or hard link would make root chown, or write, a file elsewhere. An
# existing CA is only read, so it may be a link to one (a Kubernetes secret
# projects its files as links); fix_tls_perms never chowns a linked one.
check_tls_files() {
    for f in $TLS_FILES; do
        p="${FALKORDB_TLS_PATH}/$f"
        if is_tls_ca_file "$f" && [ -e "$p" ]; then
            if [ ! -f "$p" ]; then
                echo "Error: $p is not a regular file."
                return 1
            fi
            continue
        fi
        if [ -L "$p" ] || { [ -e "$p" ] && [ -z "$(find "$p" -maxdepth 0 -type f -links 1)" ]; }; then
            echo "Error: $p is a link or not a regular file. Remove it to have it generated again."
            return 1
        fi
    done
}

# gen-certs.sh writes its keys 0600. Hand falkordb the files it generates, and
# only those, so a certificate mounted alongside is left alone: the ones an
# older image generated as root, or the ones root generated just now. A CA
# falkordb can already read stays as it is, since it is only read; one it
# cannot read is handed over only if it is a regular file with a single link.
# Returns non-zero when any of them was not handed over.
fix_tls_perms() {
    rc=0
    for f in $TLS_FILES; do
        p="${FALKORDB_TLS_PATH}/$f"
        [ -e "$p" ] || continue
        if is_tls_ca_file "$f"; then
            # shellcheck disable=SC2016 # $1 is the inner shell's
            if gosu "$FALKORDB_USER" sh -c '[ -r "$1" ]' sh "$p"; then
                continue
            fi
            if [ -L "$p" ] || [ -z "$(find "$p" -maxdepth 0 -type f -links 1)" ]; then
                echo "Warning: $FALKORDB_USER cannot read $p, and a link is not handed over."
                rc=1
                continue
            fi
        fi
        if ! chown -h "$FALKORDB_USER:$FALKORDB_USER" "$p"; then
            echo "Warning: could not hand $p to $FALKORDB_USER."
            rc=1
        fi
    done
    return $rc
}

# Called when a fix-up above failed: falkordb could not use those files, so
# redis-server stays root, as it ran before this image dropped privileges.
keep_root() {
    echo "Notice: running redis-server as root. Set SKIP_FIX_PERMS=1 to run it as $FALKORDB_USER regardless."
    drop_privs=""
}

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
    if [ "$(id -u)" = "0" ] && ! check_tls_files; then
        exit 1
    fi
    # Where falkordb can write the TLS directory, falkordb generates the files,
    # not root: anything that can write there could swap a file for a link
    # after the check above. falkordb is first handed what is already there.
    # When that fails (a read-only mount, or a CA falkordb cannot read),
    # redis-server stays root and root generates the files, as before this
    # image dropped privileges. Elsewhere, falkordb cannot plant a link, so
    # root generates the files and then hands them over. SKIP_FIX_PERMS skips
    # only the handing over.
    # shellcheck disable=SC2016 # $1 is the inner shell's
    if [ -n "$drop_privs" ] &&
        gosu "$FALKORDB_USER" sh -c '[ -w "$1" ]' sh "${FALKORDB_TLS_PATH}"; then
        if [ -z "${SKIP_FIX_PERMS:-}" ] && ! fix_tls_perms; then
            keep_root
            "${FALKORDB_BIN_PATH}/gen-certs.sh"
        else
            gosu "$FALKORDB_USER" "${FALKORDB_BIN_PATH}/gen-certs.sh"
        fi
    else
        "${FALKORDB_BIN_PATH}/gen-certs.sh"
        if [ -n "$drop_privs" ] && [ -z "${SKIP_FIX_PERMS:-}" ] && ! fix_tls_perms; then
            keep_root
        fi
    fi
fi

# Files redis writes (RDB, AOF) are private to it, as in the official image.
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
    exec "$@" redis-server ${REDIS_ARGS} --protected-mode no \
        --tls-port 6379 --port 0 \
        --tls-cert-file ${FALKORDB_TLS_PATH}/redis.crt \
        --tls-key-file ${FALKORDB_TLS_PATH}/redis.key \
        --tls-ca-cert-file ${FALKORDB_TLS_PATH}/ca.crt \
        --tls-auth-clients no \
        --dir "${FALKORDB_DATA_PATH}" \
        --loadmodule "${FALKORDB_BIN_PATH}/falkordb.so" ${FALKORDB_ARGS}
else
    # shellcheck disable=SC2086
    exec "$@" redis-server ${REDIS_ARGS} --protected-mode no \
        --dir "${FALKORDB_DATA_PATH}" \
        --loadmodule "${FALKORDB_BIN_PATH}/falkordb.so" ${FALKORDB_ARGS}
fi
