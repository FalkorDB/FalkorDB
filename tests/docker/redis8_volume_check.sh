#!/usr/bin/env bash
# Checks how a FalkorDB image starts on a data volume that Redis 8 wrote.
#
# Usage: tests/docker/redis8_volume_check.sh IMAGE
#
# The volumes are written by a released Redis image of FalkorDB (Redis 8.10.2),
# pinned by digest; REDIS8_WRITER_IMAGE overrides it. IMAGE's FALKORDB_SERVER
# picks the cases:
#   valkey: Redis 8 save files that Valkey would load -- an RDB, or the RDB
#           base of an AOF, found through the server's configuration -- stop
#           run.sh with the migration steps before valkey-server starts. A
#           Redis 8 AOF with a command base, a Valkey volume and an empty
#           volume start, and FALKORDB_REDIS_MIGRATION=1 carries out the
#           migration.
#   redis:  a Redis 8 volume starts as before, and nothing from the check
#           runs, even with FALKORDB_REDIS_MIGRATION=1.
# Exits non-zero when any case fails.
set -uo pipefail

IMAGE=${1:?usage: $0 IMAGE}
WRITER=${REDIS8_WRITER_IMAGE:-falkordb/falkordb-server:6.0.4@sha256:698a6e748b42699c1f63af01f8902652b764582e363731de56c7d566c322c209}
DATA=/var/lib/falkordb/data
PREFIX=redis8-check-$$
# Graph g1 as the writer creates it, counted by GRAPH_QUERY.
GRAPH_CREATE="UNWIND range(1, 100) AS i CREATE (:N {v: i})-[:R {w: i}]->(:M {name: 'm' + toString(i)})"
GRAPH_QUERY="MATCH (n:N)-[r:R]->(:M) RETURN count(n), sum(r.w)"
GRAPH_EXPECT="100 5050"
REFUSED="Error: this image runs Valkey"

# A CI image build pushes without loading, so the image may only be in the
# registry.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker pull -q "$IMAGE" >/dev/null || { echo "cannot pull $IMAGE" >&2; exit 2; }
fi
SERVER=$(docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$IMAGE" | sed -n 's/^FALKORDB_SERVER=//p')
case "$SERVER" in
    redis | valkey) ;;
    *) echo "FALKORDB_SERVER of $IMAGE is '$SERVER', expected redis or valkey" >&2; exit 2 ;;
esac

failures=0
# Every container and volume this run makes is named $PREFIX-*, and cleanup
# removes them by that name: many are made inside $(...), whose variables
# never reach this shell.
cleanup() {
    docker ps -aq --filter "name=^$PREFIX-" | xargs docker rm -f >/dev/null 2>&1
    docker volume ls -q --filter "name=$PREFIX-" | grep "^$PREFIX-" | xargs docker volume rm -f >/dev/null 2>&1
}
trap cleanup EXIT

fail() {
    echo "FAIL [$CASE]: $*" >&2
    failures=$((failures + 1))
}
pass() { echo "ok   [$CASE]: $*"; }

# vol NAME: a fresh Docker volume; prints its name.
vol() {
    docker volume create "$PREFIX-$1" >/dev/null
    echo "$PREFIX-$1"
}

# in_vol VOL SCRIPT: run sh SCRIPT as root with VOL at /data (and the config
# volume at /conf), in a throwaway container. These helpers run in the
# writer's image, whose tools are known, rather than in the image under test.
in_vol() {
    docker run --rm --user 0 --entrypoint sh -v "$1:/data" -v "$CONF:/conf" "$WRITER" -c "$2"
}

# copy_vol SRC NAME: a new volume holding a copy of SRC; prints its name.
copy_vol() {
    _dst=$(vol "copy-$2")
    docker run --rm --user 0 --entrypoint sh -v "$1:/src:ro" -v "$_dst:/dst" "$WRITER" -c 'cp -a /src/. /dst/'
    echo "$_dst"
}

# start NAME IMG VOL [docker run args...]: start IMG on VOL, detached. VOL is
# mounted at $MOUNT, the image's data path unless set.
start() {
    _name=$PREFIX-$1 _img=$2 _vol=$3
    shift 3
    docker rm -f "$_name" >/dev/null 2>&1
    docker run -d --name "$_name" -v "$_vol:${MOUNT:-$DATA}" -v "$CONF:/conf" "$@" "$_img" >/dev/null
}

# wait_up NAME: 0 once the server answers PING, 1 once the container has
# exited, 2 on a timeout.
wait_up() {
    for _ in $(seq 1 60); do
        if [ "$(docker inspect -f '{{.State.Running}}' "$PREFIX-$1" 2>/dev/null)" != true ]; then
            return 1
        fi
        if [ "$(docker exec "$PREFIX-$1" redis-cli PING 2>/dev/null)" = PONG ]; then
            return 0
        fi
        sleep 0.5
    done
    return 2
}

cli() {
    _name=$PREFIX-$1
    shift
    docker exec "$_name" redis-cli "$@"
}

# graph NAME: g1's node count and weight sum, space-separated.
graph() { cli "$1" GRAPH.QUERY g1 "$GRAPH_QUERY" | sed -n 3,4p | tr '\n' ' ' | sed 's/ $//'; }

stop() { docker stop -t 30 "$PREFIX-$1" >/dev/null; }
logs() { docker logs "$PREFIX-$1" 2>&1; }

# wait_rewrite NAME: wait for a BGREWRITEAOF to finish.
wait_rewrite() {
    for _ in $(seq 1 60); do
        if cli "$1" INFO persistence | tr -d '\r' | grep -qx 'aof_rewrite_in_progress:0' &&
            cli "$1" INFO persistence | tr -d '\r' | grep -qx 'aof_rewrite_scheduled:0'; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# write_redis8 NAME [REDIS_ARGS] [rewrite]: a volume the writer filled with
# graphs g1 and g2 (and their telemetry{} streams) and a string key, then
# stopped. `rewrite` rewrites the AOF first, so its base holds the data, and
# then removes the dump.rdb the writer saves on shutdown: an AOF case must
# not pass by finding that file instead.
write_redis8() {
    _vol=$(vol "$1")
    start "w-$1" "$WRITER" "$_vol" -e "REDIS_ARGS=${2:-}"
    if ! wait_up "w-$1"; then
        logs "w-$1" >&2
        echo "the writer did not start" >&2
        exit 2
    fi
    cli "w-$1" GRAPH.QUERY g1 "$GRAPH_CREATE" >/dev/null
    cli "w-$1" GRAPH.QUERY g2 "CREATE (:P {s: 'hello'})" >/dev/null
    cli "w-$1" SET plain v >/dev/null
    if [ "${3:-}" = rewrite ]; then
        cli "w-$1" BGREWRITEAOF >/dev/null
        wait_rewrite "w-$1" || { echo "the writer's AOF rewrite did not finish" >&2; exit 2; }
    fi
    stop "w-$1"
    if [ "${3:-}" = rewrite ]; then
        in_vol "$_vol" 'rm /data/dump.rdb'
    fi
    echo "$_vol"
}

# migration_ready VOL [REDIS_ARGS]: step 2 of docs/valkey-migration.md on the
# Redis writer: no telemetry, telemetry{} streams deleted, saved.
migration_ready() {
    start "m-$1" "$WRITER" "$1" -e "REDIS_ARGS=${2:-}"
    wait_up "m-$1" || { logs "m-$1" >&2; echo "the writer did not restart" >&2; exit 2; }
    cli "m-$1" GRAPH.CONFIG SET CMD_INFO no >/dev/null
    for _key in $(cli "m-$1" --scan --pattern 'telemetry{*}'); do
        cli "m-$1" DEL "$_key" >/dev/null
    done
    cli "m-$1" SAVE >/dev/null
    if [ "${3:-}" = aof ]; then
        cli "m-$1" BGREWRITEAOF >/dev/null
        wait_rewrite "m-$1" || { echo "the writer's AOF rewrite did not finish" >&2; exit 2; }
    fi
    stop "m-$1"
}

# digest VOL: one line per file under /data with its checksum.
digest() {
    _sums=$(in_vol "$1" 'cd /data && find . -type f -exec md5sum {} + | sort') && [ -n "$_sums" ] && echo "$_sums"
}

# refused NAME IMG VOL [docker run args...]: start IMG and check that run.sh
# stopped it with the message holding $EXPECT (the migration steps unless
# set), that valkey-server never started, and that the volume is unchanged.
refused() {
    _before=$(digest "$3") || { fail "cannot read the volume"; return 1; }
    start "$@"
    wait_up "$1"
    _rc=$?
    _log=$(logs "$1")
    if [ "$_rc" -ne 1 ]; then
        fail "the container is still running; expected run.sh to stop it"
        return
    fi
    _exit=$(docker inspect -f '{{.State.ExitCode}}' "$PREFIX-$1")
    if [ "$_exit" = 0 ]; then
        fail "exit code 0; expected non-zero"
    elif ! grep -qF "$REFUSED" <<<"$_log"; then
        fail "exit code $_exit without run.sh's message. Log:"$'\n'"$_log"
    elif ! grep -qF "${EXPECT:-migrate the data to Valkey}" <<<"$_log"; then
        fail "run.sh's message lacks '${EXPECT:-migrate the data to Valkey}'. Log:"$'\n'"$_log"
    elif grep -qE 'Valkey is starting|Server initialized' <<<"$_log"; then
        fail "valkey-server started before the check stopped it. Log:"$'\n'"$_log"
    elif [ "$(digest "$3")" != "$_before" ]; then
        fail "the volume changed"
    else
        pass "refused (exit $_exit): $(grep -F '(Redis RDB version' <<<"$_log" | sed 's/^ *//' | tr '\n' ' ')"
        return 0
    fi
    return 1
}

# started NAME VOL [docker run args...]: start IMAGE and check that the server
# came up and holds g1 as the writer created it.
started() {
    start "$@"
    if ! wait_up "$1"; then
        fail "the server did not start. Log:"$'\n'"$(logs "$1")"
        return 1
    fi
    _got=$(graph "$1")
    if [ "$_got" != "$GRAPH_EXPECT" ]; then
        fail "g1 holds '$_got', expected '$GRAPH_EXPECT'"
        return 1
    fi
    pass "started with g1 = $_got"
}

CONF=$(vol conf)
in_vol "$CONF" 'mkdir -p /conf/sub && chmod -R a+rX /conf'

if [ "$SERVER" = valkey ]; then
    echo "== Valkey image $IMAGE; volumes written by $WRITER"
    RDB=$(write_redis8 rdb) || exit 2
    AOF=$(write_redis8 aof "--appendonly yes" rewrite) || exit 2
    AOF_CMDS=$(write_redis8 aof-cmds "--appendonly yes --aof-use-rdb-preamble no") || exit 2
    in_vol "$CONF" 'printf "appendonly yes\nappendfilename \"aof file.aof\"\n" > /conf/quoted.conf'
    AOF_QUOTED=$(write_redis8 aof-quoted /conf/quoted.conf rewrite) || exit 2
    # dbfilename comes from a file that a relative include names, which
    # resolves against the `dir` set just before it, as the server's chdir.
    in_vol "$CONF" 'printf "dir /conf/sub\ninclude persistence.conf\n" > /conf/include.conf &&
        printf "dbfilename \"graph data.rdb\"\n" > /conf/sub/persistence.conf'
    INCLUDE=$(write_redis8 include /conf/include.conf) || exit 2
    # AOF settings from a glob include, one per file, with a non-default
    # appenddirname.
    in_vol "$CONF" 'mkdir -p /conf/parts && printf "include /conf/parts/*.conf\n" > /conf/glob.conf &&
        printf "appendonly yes\n" > /conf/parts/a.conf && printf "appenddirname aof-dir\n" > /conf/parts/b.conf'
    GLOB=$(write_redis8 glob /conf/glob.conf rewrite) || exit 2

    CASE=rdb
    refused "$CASE" "$IMAGE" "$(copy_vol "$RDB" "$CASE")"
    CASE=rdb-tls
    refused "$CASE" "$IMAGE" "$(copy_vol "$RDB" "$CASE")" -e TLS=1
    CASE=rdb-data-path
    MOUNT=/srv/graphs refused "$CASE" "$IMAGE" "$(copy_vol "$RDB" "$CASE")" -e FALKORDB_DATA_PATH=/srv/graphs &&
        { grep -qF "/srv/graphs/dump.rdb (Redis RDB version" <<<"$(logs "$CASE")" ||
            fail "the message does not name /srv/graphs/dump.rdb"; }
    CASE=aof-rdb-base
    refused "$CASE" "$IMAGE" "$(copy_vol "$AOF" "$CASE")" -e "VALKEY_ARGS=--appendonly yes"
    CASE=aof-rdb-base-redis-args
    refused "$CASE" "$IMAGE" "$(copy_vol "$AOF" "$CASE")" -e "REDIS_ARGS=--AppendOnly YES"
    CASE=aof-quoted-filename
    refused "$CASE" "$IMAGE" "$(copy_vol "$AOF_QUOTED" "$CASE")" -e "VALKEY_ARGS=/conf/quoted.conf"
    CASE=include-dir-dbfilename
    refused "$CASE" "$IMAGE" "$(copy_vol "$INCLUDE" "$CASE")" -e "VALKEY_ARGS=/conf/include.conf"
    CASE=cmdline-dir-include
    refused "$CASE" "$IMAGE" "$(copy_vol "$INCLUDE" "$CASE")" -e "VALKEY_ARGS=--dir /conf/sub --include persistence.conf"
    CASE=glob-include-appenddirname
    refused "$CASE" "$IMAGE" "$(copy_vol "$GLOB" "$CASE")" -e "VALKEY_ARGS=/conf/glob.conf" &&
        { grep -qF "$DATA/aof-dir/appendonly.aof." <<<"$(logs "$CASE")" ||
            fail "the message does not name the base file under aof-dir"; }
    CASE=include-overridden
    # A later --dbfilename wins over the include: the Redis file is not loaded.
    started_vol=$(copy_vol "$INCLUDE" "$CASE")
    start "$CASE" "$IMAGE" "$started_vol" -e "VALKEY_ARGS=/conf/include.conf --dbfilename other.rdb"
    if wait_up "$CASE"; then pass "started"; else fail "did not start. Log:"$'\n'"$(logs "$CASE")"; fi

    CASE=newer-than-valkey-reads
    v=$(copy_vol "$RDB" "$CASE")
    in_vol "$v" 'printf "REDIS0080" | dd of=/data/dump.rdb conv=notrunc 2>/dev/null'
    EXPECT="cannot load, even with" refused "$CASE" "$IMAGE" "$v" -e FALKORDB_REDIS_MIGRATION=1

    # RDB version 11, as Redis 7.2 and Valkey 8 write it, is Valkey's own: an
    # empty database (header, EOF opcode, a zero checksum) starts.
    CASE=rdb-version-11
    v=$(vol "$CASE")
    in_vol "$v" 'printf "REDIS0011\377\0\0\0\0\0\0\0\0" > /data/dump.rdb'
    start "$CASE" "$IMAGE" "$v"
    if wait_up "$CASE"; then pass "started"; else fail "did not start. Log:"$'\n'"$(logs "$CASE")"; fi

    # A Redis 8 AOF whose base holds commands, not an RDB (the empty base made
    # at startup; the graphs are in the incr file): Valkey replays it. The
    # Redis dump.rdb the writer saved next to it is not loaded with AOF on.
    CASE=aof-command-base
    v=$(copy_vol "$AOF_CMDS" "$CASE")
    if [ "$(in_vol "$v" 'head -c 6 /data/dump.rdb')" != REDIS0 ]; then
        fail "the volume has no Redis dump.rdb next to the AOF"
    fi
    started "$CASE" "$IMAGE" "$v" -e "VALKEY_ARGS=--appendonly yes"

    CASE=empty-then-valkey-volume
    v=$(vol "$CASE")
    start "$CASE" "$IMAGE" "$v"
    if ! wait_up "$CASE"; then
        fail "an empty volume did not start. Log:"$'\n'"$(logs "$CASE")"
    else
        if [ "$(cli "$CASE" CONFIG GET rdb-version-check | sed -n 2p)" != strict ]; then
            fail "rdb-version-check is not strict without the opt-in"
        else
            pass "started, rdb-version-check strict"
        fi
        cli "$CASE" GRAPH.QUERY g1 "$GRAPH_CREATE" >/dev/null
        stop "$CASE"
        CASE=valkey-volume
        started "$CASE" "$IMAGE" "$v" &&
            { [ "$(in_vol "$v" 'head -c 6 /data/dump.rdb')" = VALKEY ] ||
                fail "dump.rdb is not a Valkey RDB"; }
    fi

    # FALKORDB_REDIS_MIGRATION=1 on a volume prepared as the doc says: Valkey
    # loads it, saves it in its own format, and then starts without the flag.
    CASE=migrate-rdb
    v=$(copy_vol "$RDB" "$CASE")
    migration_ready "$v"
    if started "$CASE" "$IMAGE" "$v" -e FALKORDB_REDIS_MIGRATION=1; then
        [ "$(cli "$CASE" CONFIG GET rdb-version-check | sed -n 2p)" = relaxed ] ||
            fail "FALKORDB_REDIS_MIGRATION=1 did not relax rdb-version-check"
        cli "$CASE" SAVE >/dev/null
        stop "$CASE"
        CASE=migrate-rdb-after
        started "$CASE" "$IMAGE" "$v" &&
            { [ "$(cli "$CASE" CONFIG GET rdb-version-check | sed -n 2p)" = strict ] ||
                fail "after the migration, rdb-version-check is not strict"; }
    fi
    # TLS=1 execs through its own line: it must carry the relaxed check too.
    # The server listens on TLS only, with the certificate run.sh generates.
    # (/proc/1/cmdline cannot show the arguments: the server rewrites its
    # process title.)
    CASE=migrate-rdb-tls
    v=$(copy_vol "$RDB" "$CASE")
    migration_ready "$v"
    start "$CASE" "$IMAGE" "$v" -e FALKORDB_REDIS_MIGRATION=1 -e TLS=1
    tls_cli() { docker exec "$PREFIX-$CASE" sh -c 'redis-cli --tls --cacert "$FALKORDB_TLS_PATH/ca.crt" "$@"' sh "$@"; }
    # gen-certs.sh generates 2048-bit DH parameters before the server starts,
    # which took over 30 s on a CI arm64 runner: allow up to 5 minutes.
    _check=""
    for _ in $(seq 1 600); do
        _check=$(tls_cli CONFIG GET rdb-version-check 2>/dev/null | sed -n 2p)
        [ -n "$_check" ] && break
        [ "$(docker inspect -f '{{.State.Running}}' "$PREFIX-$CASE")" = true ] || break
        sleep 0.5
    done
    _got=$(tls_cli GRAPH.QUERY g1 "$GRAPH_QUERY" 2>/dev/null | sed -n 3,4p | tr '\n' ' ' | sed 's/ $//')
    if [ "$_check" != relaxed ]; then
        fail "rdb-version-check over TLS is '$_check', expected relaxed. Log:"$'\n'"$(logs "$CASE")"
    elif [ "$_got" != "$GRAPH_EXPECT" ]; then
        fail "g1 over TLS holds '$_got', expected '$GRAPH_EXPECT'"
    else
        pass "started on TLS, rdb-version-check relaxed, g1 = $_got"
    fi
    CASE=migrate-aof
    v=$(copy_vol "$AOF" "$CASE")
    migration_ready "$v" "--appendonly yes" aof
    if started "$CASE" "$IMAGE" "$v" -e FALKORDB_REDIS_MIGRATION=1 -e "VALKEY_ARGS=--appendonly yes"; then
        cli "$CASE" SAVE >/dev/null
        cli "$CASE" BGREWRITEAOF >/dev/null
        wait_rewrite "$CASE" || fail "BGREWRITEAOF did not finish"
        stop "$CASE"
        CASE=migrate-aof-after
        started "$CASE" "$IMAGE" "$v" -e "VALKEY_ARGS=--appendonly yes"
    fi
    CASE=user-relaxed
    v=$(copy_vol "$RDB" "$CASE")
    migration_ready "$v"
    started "$CASE" "$IMAGE" "$v" -e "VALKEY_ARGS=--rdb-version-check relaxed" &&
        { grep -qF "the Redis data check is skipped" <<<"$(logs "$CASE")" ||
            fail "no notice that the check was skipped"; }
else
    echo "== Redis image $IMAGE; volumes written by $WRITER"
    RDB=$(write_redis8 rdb) || exit 2
    AOF=$(write_redis8 aof "--appendonly yes" rewrite) || exit 2
    for CASE in rdb aof; do
        v=$(copy_vol "$([ $CASE = rdb ] && echo "$RDB" || echo "$AOF")" "$CASE")
        args=()
        [ $CASE = aof ] && args=(-e "REDIS_ARGS=--appendonly yes")
        if started "$CASE" "$IMAGE" "$v" -e FALKORDB_REDIS_MIGRATION=1 ${args[@]+"${args[@]}"}; then
            # Redis has no rdb-version-check and aborts on it, so a started
            # server also shows run.sh did not pass it.
            _log=$(logs "$CASE")
            if grep -qE 'FALKORDB_REDIS_MIGRATION|Redis data check|this image runs Valkey' <<<"$_log"; then
                fail "the Valkey check printed on a Redis image. Log:"$'\n'"$_log"
            fi
        fi
    done
fi

if [ "$failures" -ne 0 ]; then
    echo "$failures case(s) failed" >&2
    exit 1
fi
echo "All cases passed for $IMAGE ($SERVER)."
