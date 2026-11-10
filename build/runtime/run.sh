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
# files Redis 7.4 and later write: it refuses an RDB headed REDIS0 with a
# version from 12 to 79, the range it leaves to Redis, unless
# rdb-version-check is relaxed, and one above 79 even then (valkey rdb.c,
# rdbIsVersionAccepted). Redis 7.4 writes version 12, Redis 8.10 version 15.
# Before valkey-server starts, find the file it will load and, when Redis
# wrote it, stop with the migration steps. The migration is
# docs/valkey-migration.md.

# awk functions shared by the config replay and the AOF manifest reader.
SPLITARGS_AWK='
    # The arguments of line s in a[1..n], split as sds.c sdssplitargs does;
    # n, or -1 for unbalanced quotes.
    function splitargs(s, a,    n, i, len, c, d, cur, inq, insq) {
        n = 0; i = 1; len = length(s)
        for (;;) {
            while (i <= len && index(" \t\n\r\v\f", substr(s, i, 1))) i++
            if (i > len) return n
            cur = ""; inq = insq = 0
            for (;;) {
                if (i > len) {
                    if (inq || insq) return -1
                    break
                }
                c = substr(s, i, 1)
                if (inq) {
                    d = substr(s, i + 1, 1)
                    if (c == "\\" && d == "x" && substr(s, i + 2, 2) ~ /^[0-9A-Fa-f][0-9A-Fa-f]$/) {
                        cur = cur sprintf("%c", 16 * hex(substr(s, i + 2, 1)) + hex(substr(s, i + 3, 1)))
                        i += 4
                    } else if (c == "\\" && d != "") {
                        cur = cur (d == "n" ? "\n" : d == "r" ? "\r" : d == "t" ? "\t" : d == "b" ? "\b" : d == "a" ? "\007" : d)
                        i += 2
                    } else {
                        if (c == "\"") inq = 0
                        else cur = cur c
                        i++
                    }
                } else if (insq) {
                    if (c == "\\" && substr(s, i + 1, 1) == "\047") {
                        cur = cur "\047"
                        i += 2
                    } else {
                        if (c == "\047") insq = 0
                        else cur = cur c
                        i++
                    }
                } else {
                    i++
                    if (index(" \t\n\r", c)) break
                    if (c == "\"") inq = 1
                    else if (c == "\047") insq = 1
                    else cur = cur c
                }
            }
            a[++n] = cur
        }
    }
    function hex(c) { return index("0123456789abcdef", tolower(c)) - 1 }
'

# server_config NAME: the value the server will get for config NAME (lower
# case) from its arguments, empty when unset. This replays the server's config
# loading (valkey 9's config.c): a config file given as the first argument,
# then every `--name value` as a config line after it; the last setting wins.
# Lines are split into arguments, quotes and escapes included, as the server
# splits them; directive names are case-insensitive; `include` reads the files
# it names -- a glob expands -- at that point; and `dir` changes the directory
# later relative paths resolve against, as the server's chdir does. An option
# value given on the command line is taken literally, as there.
server_config() {
    _first=1
    # The tokens, in order: F <config file>, O <option name>, A <argument>.
    # shellcheck disable=SC2086
    for _tok in ${SERVER_ARGS}; do
        if [ -n "$_first" ] && [ "${_tok#-}" = "$_tok" ]; then
            printf 'F\t%s\n' "$_tok"
        elif [ "${_tok#--}" != "$_tok" ]; then
            printf 'O\t%s\n' "${_tok#--}"
        else
            printf 'A\t%s\n' "$_tok"
        fi
        _first=""
    done | SERVER_CONFIG_CWD=$PWD awk -v name="$1" "$SPLITARGS_AWK"'
        function resolve(p) { return p ~ /^\// ? p : cwd "/" p }
        function shquote(s) { gsub(/\047/, "\047\"\047\"\047", s); return "\047" s "\047" }
        # The files `include p` reads into out[1..n]; n. A glob expands,
        # sorted, from the current directory, as glob(3) does in the server.
        function expand(p, out,    n, cmd, f) {
            n = 0
            if (p !~ /[*?[]/) {
                out[++n] = resolve(p)
                return n
            }
            cmd = "cd " shquote(cwd) " && IFS= && for f in $1; do [ -e \"$f\" ] && printf \"%s\\n\" \"$f\"; done"
            cmd = "sh -c " shquote(cmd) " sh " shquote(p)
            while ((cmd | getline f) > 0) out[++n] = resolve(f)
            close(cmd)
            return n
        }
        function apply(key, arg, depth,    files, n, i) {
            key = tolower(key)
            if (key == "dir") cwd = resolve(arg)
            else if (key == "include") {
                n = expand(arg, files)
                for (i = 1; i <= n; i++) load(files[i], depth + 1)
            } else if (key == name) value = arg
        }
        # Apply config file f. A file already being read is an include cycle
        # and is skipped; depth is only a backstop against runaway nesting.
        function load(f, depth,    lines, n, i, line, a) {
            if (f in active) return
            if (depth > 100) {
                print "Warning: config includes nested more than 100 deep; the Redis data check ignores " f > "/dev/stderr"
                return
            }
            active[f] = 1
            n = 0
            while ((getline line < f) > 0) lines[++n] = line
            close(f)
            for (i = 1; i <= n; i++) {
                line = lines[i]
                sub(/^[ \t\r\n]+/, "", line)
                sub(/[ \t\r\n]+$/, "", line)
                if (line == "" || line ~ /^#/) continue
                split("", a)
                if (splitargs(line, a) == 2) apply(a[1], a[2], depth)
            }
            delete active[f]
        }
        BEGIN { cwd = ENVIRON["SERVER_CONFIG_CWD"]; value = "" }
        {
            kind = substr($0, 1, 1)
            arg = substr($0, 3)
        }
        # An option with no argument yet takes the next token, even one
        # starting with --, as its argument, as in the server -- bar --save
        # and --sentinel, which it lets stand alone.
        kind == "O" && opt != "" && !nargs && tolower(opt) !~ /^(save|sentinel)$/ { kind = "A"; arg = "--" arg }
        kind == "F" { load(resolve(arg), 0) }
        kind == "O" { opt = arg; nargs = 0 }
        kind == "A" && opt != "" && !nargs++ { apply(opt, arg, 0) }
        END { print value }
    '
}

# aof_base_file MANIFEST: two lines: the base file an AOF manifest lists (empty
# when none), then the number of base and incr records in it. Read as the
# server reads it (valkey aof.c, aofLoadManifestFromFile): `#` lines skipped,
# each line split into key/value arguments as sdssplitargs splits it, so a
# name written quoted (`file "aof file.aof.1.base.rdb" ...`) comes out whole,
# keys case-insensitive, and the base is the record whose type starts with b.
aof_base_file() {
    awk "$SPLITARGS_AWK"'
        /^#/ { next }
        {
            split("", a)
            n = splitargs($0, a)
            if (n < 6 || n % 2) next
            split("", kv)
            for (i = 1; i < n; i += 2) kv[tolower(a[i])] = a[i + 1]
            t = substr(kv["type"], 1, 1)
            if (t == "b") base = kv["file"]
            if (t == "b" || t == "i") records++
        }
        END { print base; print records + 0 }
    ' "$1"
}

# redis_rdb_version FILE: the RDB version when FILE starts with Redis's RDB
# header, REDIS0 and three digits (`REDIS0015`), as the server reads it;
# nothing otherwise. Valkey 9 heads its own files VALKEY.
redis_rdb_version() {
    _magic=$(head -c 9 "$1" 2>/dev/null) || return 0
    case "$_magic" in
        # 1NNN - 1000 rather than NNN: a leading 0 would read as octal.
        REDIS0[0-9][0-9][0-9]) echo $(( 1${_magic#REDIS0} - 1000 )) ;;
    esac
}

# check_redis_file FILE: add FILE to $redis_files when it holds a Redis RDB
# that Valkey loads only with rdb-version-check relaxed (versions 12-79), or
# to $newer_files when Valkey refuses it even then (above 79).
check_redis_file() {
    [ -f "$1" ] || return 0
    if [ ! -r "$1" ]; then
        echo "Warning: cannot read $1 to check its format."
        return 0
    fi
    _version=$(redis_rdb_version "$1")
    if [ -z "$_version" ] || [ "$_version" -lt 12 ]; then
        return 0
    elif [ "$_version" -le 79 ]; then
        redis_files="${redis_files}  $1 (Redis RDB version $_version)
"
    else
        newer_files="${newer_files}  $1 (Redis RDB version $_version)
"
    fi
}

# Before valkey-server starts, look for save files a Redis image wrote: the
# one the server will load -- the RDB file with AOF off; with AOF on, the base
# file the AOF manifest lists, as the server then loads the AOF alone. The
# rest of an AOF is commands, which Valkey replays whichever server wrote
# them. Stop with the migration steps unless the user opted in to loading
# them: FALKORDB_REDIS_MIGRATION=1, or rdb-version-check relaxed in the
# server's own arguments.
valkey_preflight() {
    redis_files=""
    newer_files=""
    # yes/no configs are case-insensitive in the server.
    if [ "$(server_config appendonly | tr '[:upper:]' '[:lower:]')" = "yes" ]; then
        _appenddirname=$(server_config appenddirname)
        _appendfilename=$(server_config appendfilename)
        _aof_dir="${FALKORDB_DATA_PATH}/${_appenddirname:-appendonlydir}"
        _manifest="${_aof_dir}/${_appendfilename:-appendonly.aof}.manifest"
        if [ -r "$_manifest" ]; then
            _info=$(aof_base_file "$_manifest")
            _base=$(printf '%s\n' "$_info" | sed -n 1p)
            _records=$(printf '%s\n' "$_info" | sed -n 2p)
            if [ -n "$_base" ]; then
                check_redis_file "${_aof_dir}/${_base}"
            fi
        else
            _base="" _records=0
        fi
        # A legacy single-file AOF in the data directory, which Valkey upgrades
        # (moves into the AOF directory) and then loads, preamble and all, when
        # the manifest is missing, lists no base or incr file, or its base
        # file is not there (aof.c, loadAppendOnlyFiles).
        if [ "${_records:-0}" -eq 0 ] || { [ -n "$_base" ] && [ ! -e "${_aof_dir}/${_base}" ]; }; then
            check_redis_file "${FALKORDB_DATA_PATH}/${_appendfilename:-appendonly.aof}"
        fi
    else
        _dbfilename=$(server_config dbfilename)
        check_redis_file "${FALKORDB_DATA_PATH}/${_dbfilename:-dump.rdb}"
    fi

    if [ -n "$newer_files" ]; then
        cat >&2 <<EOF
Error: this image runs Valkey, and ${FALKORDB_DATA_PATH} holds save files
written by a Redis version whose format Valkey cannot load, even with
FALKORDB_REDIS_MIGRATION=1:
${newer_files}
Run a Redis image of FalkorDB (a tag ending in -redis) on this volume.
See https://github.com/FalkorDB/FalkorDB/blob/main/docs/valkey-migration.md
EOF
        exit 1
    fi
    [ -n "$redis_files" ] || return 0

    if [ "${FALKORDB_REDIS_MIGRATION:-0}" = "1" ]; then
        echo "Notice: FALKORDB_REDIS_MIGRATION=1: loading save files written by Redis:"
        printf '%s' "$redis_files"
        return 0
    fi
    if [ "$(server_config rdb-version-check | tr '[:upper:]' '[:lower:]')" = "relaxed" ]; then
        echo "Notice: rdb-version-check is relaxed in the server's arguments, so the Redis data check is skipped. ${SERVER}-server will try to load save files written by Redis:"
        printf '%s' "$redis_files"
        return 0
    fi
    cat >&2 <<EOF
Error: this image runs Valkey, and ${FALKORDB_DATA_PATH} holds save files
written by Redis 7.4 or later, which Valkey refuses to load:
${redis_files}
To keep this data, run a Redis image of FalkorDB (a tag ending in -redis) on
this volume, or migrate the data to Valkey:
  1. Back up the volume: the steps below rewrite these files.
  2. On the Redis image, run GRAPH.CONFIG SET CMD_INFO no and delete every
     telemetry{*} key, and any other key Valkey cannot read as Redis saves
     it: streams, hashes with field expiry, arrays. Then run SAVE, and with
     AOF on, BGREWRITEAOF, and wait for it to finish.
  3. Start this image with FALKORDB_REDIS_MIGRATION=1. Run SAVE, and with AOF
     on, BGREWRITEAOF, and wait for it to finish.
  4. Restart this image without FALKORDB_REDIS_MIGRATION.
See https://github.com/FalkorDB/FalkorDB/blob/main/docs/valkey-migration.md
EOF
    exit 1
}

# Server arguments run.sh adds itself.
extra_args=""
if [ "$SERVER" = "valkey" ]; then
    valkey_preflight
    if [ "${FALKORDB_REDIS_MIGRATION:-0}" = "1" ]; then
        extra_args="--rdb-version-check relaxed"
        echo "Notice: FALKORDB_REDIS_MIGRATION=1: starting with --rdb-version-check relaxed. Save the data in Valkey's format (SAVE, and with AOF on, BGREWRITEAOF), then restart without it."
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
