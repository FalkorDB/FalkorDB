#!/bin/sh

# The image ships one server, named by FALKORDB_SERVER (set in the Dockerfile):
# redis or valkey. Its arguments come from REDIS_ARGS; VALKEY_ARGS is accepted
# as another name for it (both are passed when both are set).
SERVER="${FALKORDB_SERVER:-redis}"
case "$SERVER" in
    redis | valkey) ;;
    *)
        echo "Error: FALKORDB_SERVER must be redis or valkey, not '$SERVER'." >&2
        exit 1
        ;;
esac
SERVER_ARGS="${REDIS_ARGS:-} ${VALKEY_ARGS:-}"

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

# config_file_values NAME FILE [DEPTH]: every value config file FILE gives
# config NAME (lower case), one per line in the order the server applies them.
# Like the server, directive names are case-insensitive, and an `include`
# directive reads the files it names (a glob expands) at that point; DEPTH
# only bounds an include cycle.
config_file_values() {
    { [ -f "$2" ] && [ -r "$2" ] && [ "${3:-0}" -lt 8 ]; } || return 0
    awk -v name="$1" '
        { sub(/^[ \t]+/, ""); sub(/[ \t\r]+$/, "") }
        $0 == "" || /^#/ { next }
        {
            key = tolower($1)
            rest = $0
            sub(/^[^ \t]+[ \t]+/, "", rest)
        }
        key == "include" && NF == 2 { gsub(/^["\047]|["\047]$/, "", rest); print "i " rest; next }
        key == name && NF >= 2 { print "v " rest }
    ' "$2" | while IFS= read -r _line; do
        case "$_line" in
            "i "*)
                # Unquoted, so an include glob expands as in the server.
                # shellcheck disable=SC2086
                for _inc in ${_line#i }; do
                    config_file_values "$1" "$_inc" $(( ${3:-0} + 1 ))
                done
                ;;
            *) printf '%s\n' "${_line#v }" ;;
        esac
    done
}

# server_config NAME: the value the server will get for config NAME (lower
# case) from its arguments -- a config file given as the first argument and
# the files it includes, then any `--NAME value` or `--include FILE`, the last
# one winning, as in the server. Empty when unset.
server_config() {
    _value="" _want="" _first=1
    # shellcheck disable=SC2086
    for _tok in ${SERVER_ARGS}; do
        _file=""
        case "$_want" in
            value) _value=$_tok ;;
            include) _file=$_tok ;;
            *)
                if [ -n "$_first" ] && [ "${_tok#-}" = "$_tok" ]; then
                    _file=$_tok
                fi
                case "$_tok" in
                    --*)
                        case "$(printf '%s' "$_tok" | tr '[:upper:]' '[:lower:]')" in
                            "--$1") _want=value ;;
                            --include) _want=include ;;
                        esac
                        _first=""
                        continue
                        ;;
                esac
                ;;
        esac
        _want="" _first=""
        if [ -n "$_file" ]; then
            _file_value=$(config_file_values "$1" "$_file" | tail -n 1)
            [ -z "$_file_value" ] || _value=$_file_value
        fi
    done
    _value=${_value#[\"\']}
    _value=${_value%[\"\']}
    printf '%s\n' "$_value"
}

# redis_rdb_version FILE: the save-file version when FILE starts with a Redis
# RDB header (`REDIS0012`), nothing otherwise.
redis_rdb_version() {
    _magic=$(head -c 9 "$1" 2>/dev/null) || return 0
    case "$_magic" in
        # 1NNNN - 10000 rather than NNNN: a leading 0 would read as octal.
        REDIS[0-9][0-9][0-9][0-9]) echo $(( 1${_magic#REDIS} - 10000 )) ;;
    esac
}

# check_redis_file FILE: add FILE to $redis_files when it holds a Redis RDB of
# version 12 or later, as Redis 8 writes. Valkey keeps versions 12-79 for
# Redis and refuses them unless rdb-version-check is relaxed.
check_redis_file() {
    [ -f "$1" ] || return 0
    if [ ! -r "$1" ]; then
        echo "Warning: cannot read $1 to check its format."
        return 0
    fi
    _version=$(redis_rdb_version "$1")
    if [ -n "$_version" ] && [ "$_version" -ge 12 ]; then
        redis_files="${redis_files}  $1 (Redis RDB version $_version)
"
    fi
}

# Before valkey-server starts, look for save files a Redis image wrote: the
# one the server will load -- the RDB file with AOF off; with AOF on, the base
# file the AOF manifest lists, as the server then loads the AOF alone.
# Valkey would exit on them, so explain what to do instead -- unless
# FALKORDB_REDIS_MIGRATION=1 asks to load them anyway.
valkey_preflight() {
    redis_files=""
    # yes/no configs are case-insensitive in the server.
    if [ "$(server_config appendonly | tr '[:upper:]' '[:lower:]')" = "yes" ]; then
        _appenddirname=$(server_config appenddirname)
        _appendfilename=$(server_config appendfilename)
        _aof_dir="${FALKORDB_DATA_PATH}/${_appenddirname:-appendonlydir}"
        _manifest="${_aof_dir}/${_appendfilename:-appendonly.aof}.manifest"
        if [ -r "$_manifest" ]; then
            # Manifest lines are `file <name> seq <n> type <b|h|i>` key/value
            # pairs; the base file is type b.
            _base=$(awk '{ split("", kv); for (i = 1; i < NF; i += 2) kv[$i] = $(i + 1); if (kv["type"] == "b") base = kv["file"] } END { print base }' "$_manifest")
            if [ -n "$_base" ]; then
                check_redis_file "${_aof_dir}/${_base}"
            fi
        fi
    else
        _dbfilename=$(server_config dbfilename)
        check_redis_file "${FALKORDB_DATA_PATH}/${_dbfilename:-dump.rdb}"
    fi
    [ -n "$redis_files" ] || return 0

    if [ "${FALKORDB_REDIS_MIGRATION:-0}" = "1" ]; then
        echo "Notice: FALKORDB_REDIS_MIGRATION=1: loading save files written by Redis:"
        printf '%s' "$redis_files"
        return 0
    fi
    cat >&2 <<EOF
Error: this image runs Valkey, and ${FALKORDB_DATA_PATH} holds save files
written by Redis, which Valkey refuses to load:
${redis_files}
To keep using this data, either:
  * run a Redis-based FalkorDB image on this volume, or
  * migrate it to Valkey. Back the volume up first: Valkey rewrites these files
    in place. Valkey cannot read the streams or hash-field expiry that Redis 8
    saves, and FalkorDB keeps query telemetry in telemetry{<graph>} streams, so
    on the Redis image run GRAPH.CONFIG SET CMD_INFO no, delete the
    telemetry{*} keys and any other such keys, then save (BGREWRITEAOF with AOF
    on, then BGSAVE). Start this image once with FALKORDB_REDIS_MIGRATION=1,
    save again the same way so the files are rewritten in Valkey format, and
    restart without it.
EOF
    exit 1
}

# Server arguments run.sh adds itself.
extra_args=""
if [ "$SERVER" = "valkey" ]; then
    valkey_preflight
    if [ "${FALKORDB_REDIS_MIGRATION:-0}" = "1" ]; then
        extra_args="--rdb-version-check relaxed"
        echo "Notice: FALKORDB_REDIS_MIGRATION=1: starting with --rdb-version-check relaxed. Unset it once the data is saved in Valkey format."
    fi
elif [ "${FALKORDB_REDIS_MIGRATION:-0}" = "1" ]; then
    echo "Notice: FALKORDB_REDIS_MIGRATION only applies to Valkey images; ignored."
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
    exec "$@" "${SERVER}-server" ${SERVER_ARGS} ${extra_args} --protected-mode no \
        --tls-port 6379 --port 0 \
        --tls-cert-file ${FALKORDB_TLS_PATH}/redis.crt \
        --tls-key-file ${FALKORDB_TLS_PATH}/redis.key \
        --tls-ca-cert-file ${FALKORDB_TLS_PATH}/ca.crt \
        --tls-auth-clients no \
        --dir "${FALKORDB_DATA_PATH}" \
        --loadmodule "${FALKORDB_BIN_PATH}/falkordb.so" ${FALKORDB_ARGS}
else
    # shellcheck disable=SC2086
    exec "$@" "${SERVER}-server" ${SERVER_ARGS} ${extra_args} --protected-mode no \
        --dir "${FALKORDB_DATA_PATH}" \
        --loadmodule "${FALKORDB_BIN_PATH}/falkordb.so" ${FALKORDB_ARGS}
fi
