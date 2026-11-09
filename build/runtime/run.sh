#!/bin/sh

# redis-server runs as the unprivileged `falkordb` user (uid 999, created in the
# Dockerfile), as in the official redis image. The container itself still
# starts as root: the browser entrypoint needs root to switch to nextjs, and a
# data volume written by an older image is owned by root and has to be handed
# over first. Root is kept when the container was started with --user, when
# SKIP_DROP_PRIVS is set, when it lacks CAP_SETUID/CAP_SETGID (e.g.
# --cap-drop ALL) and so could not switch users anyway, or when falkordb
# could not be given the files redis-server uses. SKIP_FIX_PERMS skips the
# ownership fix-up. Both names are the official image's.
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
# and as `falkordb` the node cannot rewrite it and exits at startup. chown runs
# with -execdir, from each directory find has open, so a directory falkordb
# swaps for a link while find walks does not lead it elsewhere. find starts
# from the directory itself, not a link to it (the image's /data is one),
# whose owner chown -h would change instead.
# Returns non-zero when the dir was not handed over.
fix_data_dir_perms() {
    data_dir=$(cd -P "${FALKORDB_DATA_PATH}" && pwd -P) || return 1
    unknown=$(find "$data_dir" -mindepth 1 -maxdepth 1 \
        ! \( \( -type f \( -name '*.rdb' -o -name 'nodes*.conf' \) \) -o \
        \( -type d -name appendonlydir \) \) -print -quit)
    if [ -n "$unknown" ]; then
        echo "Notice: unknown file '$unknown' in ${FALKORDB_DATA_PATH}; ownership not changed."
        return 1
    fi
    # A fixed PATH: -execdir refuses to run with a relative or empty entry in it.
    if ! PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        find "$data_dir" ! -user "$FALKORDB_USER" \
        -execdir chown -h "$FALKORDB_USER:$FALKORDB_USER" {} +; then
        echo "Warning: could not hand ${FALKORDB_DATA_PATH} to $FALKORDB_USER."
        return 1
    fi
}

# The files gen-certs.sh generates in the TLS directory.
TLS_FILES="ca.key ca.crt ca.txt openssl.cnf server.key server.crt
           client.key client.crt redis.key redis.crt redis.dh"
# The ones gen-certs.sh writes on every start. It writes the others only when
# [ -f ] finds no regular file there, and otherwise only reads them.
TLS_REWRITTEN_FILES="ca.txt server.crt client.crt redis.crt"

# Run as root inside a TLS directory falkordb does not control, before root
# generates the files there: refuse what root would write through. A file
# rewritten on every start must be absent or a regular file with a single
# link. Any other may also be a link to a regular file (a CA or a key
# supplied to use), but not a link that leads nowhere, or something else.
# Root or the operator left it there, or falkordb did when it could still
# change the directory. Where falkordb controls the directory, root writes
# nothing there (see the TLS block below).
check_tls_files() {
    for f in $TLS_FILES; do
        case " $TLS_REWRITTEN_FILES " in
            *" $f "*)
                if [ -L "$f" ] || { [ -e "$f" ] &&
                    [ -z "$(find "$f" -maxdepth 0 -type f -links 1)" ]; }; then
                    echo "Error: ${FALKORDB_TLS_PATH}/$f is a link or not a regular file. Remove it to have it generated again."
                    return 1
                fi
                ;;
            *)
                if { [ -L "$f" ] || [ -e "$f" ]; } && [ ! -f "$f" ]; then
                    echo "Error: ${FALKORDB_TLS_PATH}/$f is a link that leads nowhere, or not a regular file."
                    return 1
                fi
                ;;
        esac
    done
}

# falkordb_can_change PATH: does falkordb own PATH, or may it write it by its
# group or other mode bits? Read by root from the owner, group and mode, since
# falkordb could make its own directory unreadable to itself for a check run
# as falkordb, and a read-only mount here may be writable for another
# container. For a link, pass -L first, to judge what it leads to: a link's
# own mode bits are always rwx.
falkordb_can_change() {
    [ -n "$(find "$@" -maxdepth 0 \( -user "$FALKORDB_USER" -o \
        \( -group "$FALKORDB_USER" -perm -g+w \) -o -perm -o+w \) -print 2>/dev/null)" ]
}

# Can falkordb change the current directory? Its mode bits, as above, or
# anything else (an ACL) that lets falkordb write it.
falkordb_controls() {
    falkordb_can_change . || gosu "$FALKORDB_USER" sh -c '[ -w . ]'
}

# enter_tls_dir PATH: cd into the TLS directory, one name at a time, following
# links by hand and creating what is missing. When root can switch to
# falkordb, it refuses to look a name up in a directory falkordb controls,
# where falkordb could have put a link, or another directory, by that name.
enter_tls_dir() {
    case "$1" in
        /*) rest=$1 ;;
        *) rest="$(pwd)/$1" ;;
    esac
    links=0
    cd / || return 1
    while [ -n "$rest" ]; do
        part=${rest%%/*}
        rest=${rest#"$part"}
        rest=${rest#/}
        if [ -z "$part" ] || [ "$part" = . ]; then
            continue
        fi
        if [ -n "$can_switch" ] && falkordb_controls; then
            echo "Error: $FALKORDB_USER can change $(pwd -P), on the way to ${FALKORDB_TLS_PATH}, so it could swap the TLS directory. Use one under directories only root can change."
            return 1
        fi
        if [ -L "./$part" ]; then
            links=$((links + 1))
            if [ "$links" -gt 40 ]; then
                echo "Error: too many links on the way to ${FALKORDB_TLS_PATH}."
                return 1
            fi
            target=$(find "./$part" -maxdepth 0 -printf '%l')
            case "$target" in
                /*) cd / || return 1 ;;
            esac
            rest="$target/$rest"
            continue
        fi
        if [ ! -e "./$part" ] && ! mkdir "./$part"; then
            echo "Error: cannot create ${FALKORDB_TLS_PATH}."
            return 1
        fi
        if ! cd -P "./$part"; then
            echo "Error: cannot enter ${FALKORDB_TLS_PATH}."
            return 1
        fi
    done
}

# The functions below run inside the TLS directory and name its files
# relative to it (see the TLS block further down).

# Run in a subshell that has NAME open on fd 3: is fd 3 the regular file, with
# a single link, that NAME is now? Then no link swapped in after the checks
# led the open elsewhere, and root acts on fd 3 rather than on the name.
fd3_is() {
    opened=$(find -L /proc/self/fd/3 -maxdepth 0 -printf '%D %i %n %y') &&
        [ "$opened" = "$(find "$1" -maxdepth 0 -printf '%D %i %n %y')" ] || return 1
    case "$opened" in
        *" 1 f") return 0 ;;
    esac
    return 1
}

# For a TLS directory falkordb controls, where root changes nothing: falkordb
# takes over, by copying, the files there it can read but does not own (an
# older image generated them as root). The CA is left as it is, since it is
# only read.
adopt_tls_files() {
    # shellcheck disable=SC2016,SC2086 # the inner shell's $f; split the list
    gosu "$FALKORDB_USER" sh -c '
        for f; do
            if [ -e "$f" ] && [ -r "$f" ] && [ ! -O "$f" ] &&
                [ "$f" != ca.key ] && [ "$f" != ca.crt ]; then
                case "$f" in
                    *.key) umask 077 ;;
                    *) umask 022 ;;
                esac
                cat "$f" > "$f.adopt" && rm -f "$f" && cat "$f.adopt" > "$f" &&
                    rm -f "$f.adopt" || echo "Warning: could not take over $f."
            fi
        done' sh $TLS_FILES
}

# Can falkordb read every TLS file there is?
falkordb_reads_tls() {
    # shellcheck disable=SC2016,SC2086 # the inner shell's $f; split the list
    gosu "$FALKORDB_USER" sh -c '
        for f; do
            [ ! -e "$f" ] || [ -r "$f" ] || exit 1
        done' sh $TLS_FILES
}

# Is $1/ca.crt a certificate for $1/ca.key's public key, signed by that key?
# Only whoever holds the key can make one: falkordb cannot when it cannot read
# the key. $1/ca.crt is first cut down to its first certificate, the one
# openssl x509 and gen-certs.sh read: with another one after it, openssl
# verify would build a chain to that one and check its signature instead.
# -check_ss_sig has the signature checked; without it, openssl accepts a
# certificate given as its own trust anchor unsigned.
ca_crt_from_ca_key() {
    openssl x509 -in "$1/ca.crt" -out "$1/ca.crt.first" 2>/dev/null &&
        cat "$1/ca.crt.first" > "$1/ca.crt" && rm -f "$1/ca.crt.first" &&
        key_pub=$(openssl pkey -in "$1/ca.key" -pubout 2>/dev/null) &&
        [ -n "$key_pub" ] &&
        [ "$(openssl x509 -in "$1/ca.crt" -noout -pubkey 2>/dev/null)" = "$key_pub" ] &&
        openssl verify -no-CApath -no-CAstore -no_check_time -check_ss_sig \
            -CAfile "$1/ca.crt" "$1/ca.crt" > /dev/null 2>&1
}

# For a TLS directory falkordb controls, when falkordb cannot read every file
# there (a CA it cannot read, say): root generates the files in a directory
# only root can reach, and falkordb copies the results in. falkordb hands
# root the files it can read. Root reads the others itself, and only regular
# files falkordb cannot change, not through links; it never copies them back.
# With a CA key falkordb cannot read, root reads ca.crt itself too, and
# accepts one falkordb can change only if that key signed it for itself (the
# one an earlier start generated, which falkordb copied in). Root always
# writes ca.txt and openssl.cnf afresh. So falkordb chooses nothing about
# what root signs but its own keys. An existing openssl.cnf is left as it is.
gen_certs_staged() {
    # /dev/shm when /tmp is read-only (docker run --read-only)
    staging=$(mktemp -d /tmp/falkordb-tls.XXXXXX 2>/dev/null ||
        mktemp -d /dev/shm/falkordb-tls.XXXXXX) || {
        echo "Error: no temporary directory to generate the TLS files in. Mount a tmpfs at /tmp."
        exit 1
    }
    # What is not copied back, decided before anything in the directory can
    # change: every file there now but the certificates and ca.txt, which
    # gen-certs.sh writes again (it only creates the others when missing), and
    # whatever root read.
    kept=""
    root_ca=""
    for f in $TLS_FILES; do
        if [ ! -e "$f" ] || [ "$f" = ca.txt ]; then
            continue
        fi
        case "$f" in
            server.crt | client.crt | redis.crt) ;;
            *) kept="$kept $f" ;;
        esac
        [ "$f" != openssl.cnf ] || continue
        # shellcheck disable=SC2016 # $1 is the inner shell's
        if { [ "$f" != ca.crt ] || [ -z "$root_ca" ]; } &&
            gosu "$FALKORDB_USER" sh -c '[ -r "$1" ]' sh "$f"; then
            gosu "$FALKORDB_USER" cat "$f" > "$staging/$f"
            continue
        fi
        echo "Notice: root reads ${FALKORDB_TLS_PATH}/$f itself, to sign the certificates; it is not copied back."
        # 0: read, and falkordb cannot change it; 10: read, but falkordb can.
        # Anything else: not read.
        read_rc=1
        if [ -f "$f" ]; then
            (exec 3< "$f" && fd3_is "$f" && cat <&3 > "$staging/$f" || exit 1
                ! falkordb_can_change -L /proc/self/fd/3 || exit 10)
            read_rc=$?
        fi
        if [ "$read_rc" -eq 10 ] && [ "$f" = ca.crt ] && [ -n "$root_ca" ] &&
            ca_crt_from_ca_key "$staging"; then
            read_rc=0
        fi
        if [ "$read_rc" -ne 0 ]; then
            if [ "$f" = ca.crt ] && [ -n "$root_ca" ]; then
                echo "Error: root reads ${FALKORDB_TLS_PATH}/ca.crt only as a regular file, not through a link, that $FALKORDB_USER cannot change or that ca.key signed for itself. Make it a file only root can change."
            else
                echo "Error: root reads ${FALKORDB_TLS_PATH}/$f only as a regular file that $FALKORDB_USER cannot change, not through a link. Make it readable by $FALKORDB_USER, or a file only root can change."
            fi
            rm -rf "$staging"
            exit 1
        fi
        kept="$kept $f"
        if [ "$f" = ca.key ]; then
            root_ca=1
        fi
    done
    FALKORDB_TLS_PATH="$staging" "${FALKORDB_BIN_PATH}/gen-certs.sh"
    for f in $TLS_FILES; do
        case "$kept " in
            *" $f "*) continue ;;
        esac
        [ -f "$staging/$f" ] || continue
        case "$f" in
            *.key) mask=077 ;;
            *) mask=022 ;;
        esac
        # Not mv: on Debian it loads libacl1, which the VEX statements say
        # nothing here runs.
        # shellcheck disable=SC2016 # $1 and $2 are the inner shell's
        if ! gosu "$FALKORDB_USER" sh -c 'umask "$2" && rm -f "$1" && cat > "$1"' \
            sh "$f" "$mask" < "$staging/$f"; then
            echo "Warning: $FALKORDB_USER could not write ${FALKORDB_TLS_PATH}/$f."
        fi
    done
    rm -rf "$staging"
}

# For a TLS directory falkordb does not control. gen-certs.sh writes its keys
# 0600. Hand falkordb the keys and certificates it generates, and only those,
# so a certificate mounted alongside is left alone: the ones an older image
# generated as root, or the ones root generated just now. Not the CA files,
# ca.txt or openssl.cnf: root signs with them here, and falkordb must not
# choose what root signs. Returns non-zero when any of them was not handed
# over.
fix_tls_perms() {
    rc=0
    for f in $TLS_FILES; do
        case "$f" in
            ca.key | ca.crt | ca.txt | openssl.cnf) continue ;;
        esac
        [ -e "$f" ] || continue
        if ! (exec 3< "$f" && fd3_is "$f" &&
            chown "$FALKORDB_USER:$FALKORDB_USER" /proc/self/fd/3); then
            echo "Warning: could not hand ${FALKORDB_TLS_PATH}/$f to $FALKORDB_USER."
            rc=1
        fi
    done
    return $rc
}

# Called when a fix-up above failed: falkordb could not use those files, so
# redis-server stays root, as it ran before this image dropped privileges.
keep_root() {
    [ -n "$drop_privs" ] || return 0
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

# drop_privs: redis-server runs as falkordb. can_switch: root may switch to
# falkordb at all, which keep_root does not change.
drop_privs=""
can_switch=""
if [ "$(id -u)" = "0" ] && [ -z "${SKIP_DROP_PRIVS:-}" ] && has_cap 6 && has_cap 7; then
    drop_privs=1
    can_switch=1
    if [ -z "${SKIP_FIX_PERMS:-}" ] && ! fix_data_dir_perms; then
        keep_root
    fi
fi

if [ "${TLS:-0}" -eq "1" ]; then
    if [ "$(id -u)" = "0" ]; then
        # Root works inside the TLS directory, so every name it checks,
        # writes or chowns is looked up in that one directory, even if its
        # path is moved or replaced meanwhile.
        run_dir=$(pwd)
        if ! enter_tls_dir "${FALKORDB_TLS_PATH}"; then
            exit 1
        fi
    fi
    # Where falkordb controls the TLS directory, root changes nothing there,
    # and reads there only through fd3_is: falkordb could swap a file for a
    # link, or move another file in, while root works. That holds when
    # redis-server stays root for the data directory too. A link falkordb
    # planted there leads only its own gen-certs.sh, with its own rights.
    # falkordb takes over the files it can read and generates the rest. When
    # it cannot read them all (a CA it cannot read, say), root generates them
    # elsewhere and falkordb copies them in. Where falkordb does not control
    # the directory, root refuses links it would write through
    # (check_tls_files), generates the files there and then hands them over.
    # SKIP_FIX_PERMS skips the taking and handing over. When root cannot
    # switch users at all, whoever runs redis-server generates the files in
    # place, as before this image dropped privileges.
    if [ -n "$can_switch" ] && falkordb_controls; then
        if [ -z "${SKIP_FIX_PERMS:-}" ]; then
            adopt_tls_files
        fi
        if falkordb_reads_tls; then
            FALKORDB_TLS_PATH=. gosu "$FALKORDB_USER" "${FALKORDB_BIN_PATH}/gen-certs.sh"
        else
            gen_certs_staged
        fi
    elif [ "$(id -u)" = "0" ]; then
        if [ -n "$can_switch" ]; then
            # What root signs with that falkordb could change (an older image
            # handed it over) is written afresh, or refused: falkordb must not
            # choose what root signs.
            for f in openssl.cnf ca.txt; do
                # shellcheck disable=SC2016 # $1 is the inner shell's
                if [ -e "$f" ] && { falkordb_can_change -L "$f" ||
                    gosu "$FALKORDB_USER" sh -c '[ -w "$1" ]' sh "$f"; }; then
                    rm -f "$f"
                fi
            done
            if ! check_tls_files; then
                exit 1
            fi
            # shellcheck disable=SC2016 # $1 is the inner shell's
            if [ -e ca.key ] && [ -e ca.crt ] && falkordb_can_change -L ca.crt &&
                ! gosu "$FALKORDB_USER" sh -c '[ -r "$1" ]' sh ca.key; then
                echo "Error: $FALKORDB_USER can change ${FALKORDB_TLS_PATH}/ca.crt but cannot read ca.key, so root does not sign with them. Make ca.crt a file only root can change."
                exit 1
            fi
        fi
        FALKORDB_TLS_PATH=. "${FALKORDB_BIN_PATH}/gen-certs.sh"
        if [ -n "$drop_privs" ] && [ -z "${SKIP_FIX_PERMS:-}" ] && ! fix_tls_perms; then
            keep_root
        fi
    else
        "${FALKORDB_BIN_PATH}/gen-certs.sh"
    fi
    if [ "$(id -u)" = "0" ]; then
        cd "$run_dir" || exit 1
    fi
    # redis-server stays root if falkordb cannot read what it loads, by the
    # paths it opens them by. SKIP_FIX_PERMS runs it as falkordb regardless.
    if [ -n "$drop_privs" ] && [ -z "${SKIP_FIX_PERMS:-}" ]; then
        for f in redis.crt redis.key ca.crt; do
            # shellcheck disable=SC2016 # $1 is the inner shell's
            if ! gosu "$FALKORDB_USER" sh -c '[ -r "$1" ]' sh "${FALKORDB_TLS_PATH}/$f"; then
                echo "Warning: $FALKORDB_USER cannot read ${FALKORDB_TLS_PATH}/$f."
                keep_root
                break
            fi
        done
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
