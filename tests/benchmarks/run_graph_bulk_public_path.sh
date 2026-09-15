#!/usr/bin/env bash
# Prepare a pinned Redis server in this worktree and run the public GRAPH.BULK
# benchmark.  REDIS_SERVER may override the pinned server for local debugging.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly redis_version="8.6.3"
readonly redis_root="$root/bin/perfloop-redis-$redis_version"
readonly redis_source="$redis_root/source"
readonly redis_server="$redis_source/src/redis-server"

log() {
	echo "$*" >&2
}

redis_major_version() {
	local version
	version="$($1 --version 2>/dev/null || true)"
	if [[ "$version" =~ v=([0-9]+)\. ]]; then
		echo "${BASH_REMATCH[1]}"
		return 0
	fi
	return 1
}

require_redis_8() {
	local major
	major="$(redis_major_version "$1")" || {
		log "Unable to determine Redis version for $1"
		return 1
	}
	if (( major < 8 )); then
		log "GRAPH.BULK requires Redis 8+, but $1 reports Redis $major"
		return 1
	fi
}

build_redis() {
	if [[ -x "$redis_server" ]]; then
		require_redis_8 "$redis_server"
		echo "$redis_server"
		return 0
	fi

	local archive jobs downloader
	archive="$redis_root/redis-$redis_version.tar.gz"
	jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
	mkdir -p "$redis_root"

	if [[ ! -f "$archive" ]]; then
		log "Downloading Redis $redis_version for GRAPH.BULK tests"
		if command -v curl >/dev/null 2>&1; then
			downloader=(curl --fail --location --silent --show-error
				"https://github.com/redis/redis/archive/refs/tags/$redis_version.tar.gz"
				--output "$archive")
		elif command -v wget >/dev/null 2>&1; then
			downloader=(wget --quiet
				"https://github.com/redis/redis/archive/refs/tags/$redis_version.tar.gz"
				--output-document "$archive")
		else
			log "Neither curl nor wget is available to download Redis $redis_version"
			return 1
		fi
		"${downloader[@]}"
	fi

	log "Building Redis $redis_version for GRAPH.BULK tests"
	rm -rf "$redis_source"
	mkdir -p "$redis_source"
	tar -xzf "$archive" --strip-components=1 -C "$redis_source"
	make -C "$redis_source" -j"$jobs" >&2
	require_redis_8 "$redis_server"
	echo "$redis_server"
}

if [[ "${1:-}" == "--prepare" ]]; then
	if [[ -n "${REDIS_SERVER:-}" ]]; then
		require_redis_8 "$REDIS_SERVER"
	else
		build_redis >/dev/null
	fi
	exit 0
fi

if [[ -n "${REDIS_SERVER:-}" ]]; then
	require_redis_8 "$REDIS_SERVER"
	selected_redis="$REDIS_SERVER"
else
	selected_redis="$(build_redis)"
fi

module="${MODULE:-$root/bin/linux-x64-release/falkordb.so}"
exec python3 "$root/tests/benchmarks/graph_bulk_public_path.py" \
	--module "$module" --redis-server "$selected_redis" "$@"
