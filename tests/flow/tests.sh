#!/bin/bash

# [[ $VERBOSE == 1 ]] && set -x

PROGNAME="${BASH_SOURCE[0]}"
HERE="$(cd "$(dirname "$PROGNAME")" &>/dev/null && pwd)"
ROOT=$(cd $HERE/../.. && pwd)

# Source common definitions and functions
. "$ROOT/tests/common.sh"

export PYTHONUNBUFFERED=1

# Redis version for testing (used for reference)
REDIS_VER=8.0

cd $HERE

#----------------------------------------------------------------------------------------------

help() {
	cat <<-END
		Run flow tests

		[ARGVARS...] tests.sh [--help|help]

		Argument variables:
		MODULE=path           Module .so path

		TEST=name             Run specific test (e.g. test.py:test_name)
		TESTFILE=file         Run tests listed in `file`
		FAILEDFILE=file       Write failed tests into `file`

		GEN=1                 General tests on standalone Redis (default)
		AOF=1                 AOF persistency tests on standalone Redis
		TCK=1                 Cypher Technology Compatibility Kit tests
		UPGRADE=1             Upgrade tests
		REDIS_SERVER=path     Location of redis-server
		REDIS_PORT=n          Redis server port

		EXT=1|run             Test on existing env (1=running; run=start redis-server)
		EXT_HOST=addr         Address of existing env (default: 127.0.0.1)
		EXT_PORT=n            Port of existing env

		RLEC=0|1              General tests on RLEC
		DOCKER_HOST=addr      Address of Docker server (default: localhost)
		RLEC_PORT=port        Port of RLEC database (default: 12000)

		COV=1                 Run with coverage analysis
		VG=1                  Run with Valgrind
		VG_LEAKS=1            Look for memory leaks
		VG_ACCESS=1           Look for memory access errors
		SAN=type              Use LLVM sanitizer (type=address|memory|leak|thread)
		BB=1                  Enable Python debugger (break using BB() in tests)
		GDB=1                 Enable interactive gdb debugging (in single-test mode)

		RLTEST=path|'view'    Take RLTest from repo path or from local view
		RLTEST_DEBUG=1        Show debugging printouts from tests
		RLTEST_ARGS=args      Extra RLTest args

		PARALLEL=1            Runs tests in parallel
		SLOW=1                Do not test in parallel
		UNIX=1                Use unix sockets
		RANDPORTS=1           Use randomized ports

		CLEAR_LOGS=0          Do not remove logs prior to running tests

		LIST=1                List all tests and exit
		ENV_ONLY=1            Just start environment, run no tests
		V|VERBOSE=1           Print commands and Redis output
		LOG=1                 Send results to log (even on single-test mode)
		KEEP=1                Do not remove intermediate files
		NOP=1                 Dry run
		HELP=1                Show help

	END
}

#----------------------------------------------------------------------------------------------

traps() {
	local func="$1"
	shift
	local sig
	for sig in "$@"; do
		trap "$func $sig" "$sig"
	done
}

linux_stop() {
	local pgid=$(cat /proc/$PID/status | grep pgid | awk '{print $2}')
	kill -9 -- -$pgid
}

macos_stop() {
	local pgid=$(ps -o pid,pgid -p $PID | awk "/$PID/"'{ print $2 }' | tail -1)
	pkill -9 -g $pgid
}

stop() {
	trap - SIGINT
	if [[ $OS == linux ]]; then
		linux_stop
	elif [[ $OS == macos ]]; then
		macos_stop
	fi
	exit 1
}

traps 'stop' SIGINT

#----------------------------------------------------------------------------------------------

setup_rltest() {
	if [[ $RLTEST == view ]]; then
		if [[ ! -d $ROOT/../RLTest ]]; then
			eprint "RLTest not found in view $ROOT"
			exit 1
		fi
		RLTEST=$(cd $ROOT/../RLTest; pwd)
	fi

	if [[ -n $RLTEST ]]; then
		if [[ ! -d $RLTEST ]]; then
			eprint "Invalid RLTest location: $RLTEST"
			exit 1
		fi

		# Specifically search for it in the specified location
		export PYTHONPATH="$PYTHONPATH:$RLTEST"
		if [[ $VERBOSE == 1 ]]; then
			echo "PYTHONPATH=$PYTHONPATH"
		fi
	fi

	RLTEST_ARGS+=" --enable-debug-command --no-progress"

	if [[ $RLTEST_VERBOSE == 1 ]]; then
		RLTEST_ARGS+=" -v"
	fi
	if [[ $RLTEST_DEBUG == 1 ]]; then
		RLTEST_ARGS+=" --debug-print"
	fi
	if [[ -n $RLTEST_LOG && $RLTEST_LOG != 1 ]]; then
		RLTEST_ARGS+=" -s"
	fi
	if [[ $RLTEST_CONSOLE == 1 ]]; then
		RLTEST_ARGS+=" -i"
	fi
}

#----------------------------------------------------------------------------------------------

build_redis_with_sanitizer() {
	local san_type=$1
	local redis_dir="/tmp/redis-san-build"
	local redis_version="8.0.0"
	local ignorelist=$ROOT/tests/memcheck/redis.san-ignorelist
	local build_log="/tmp/redis-san-build.log"

	echo "Building Redis $redis_version with $san_type sanitizer..."

	# Download and extract Redis if not already present
	if [[ ! -d "$redis_dir" ]]; then
		mkdir -p "$redis_dir"
		cd /tmp
		if [[ ! -f "redis-$redis_version.tar.gz" ]]; then
			echo "Downloading Redis $redis_version..."
			wget -q "https://github.com/redis/redis/archive/refs/tags/$redis_version.tar.gz" -O "redis-$redis_version.tar.gz"
		fi
		tar xzf "redis-$redis_version.tar.gz" -C "$redis_dir" --strip-components=1
	fi

	cd "$redis_dir"

	# Clean previous build
	make distclean > /dev/null 2>&1 || true

	# Build with sanitizer flags
	local build_result=0
	if [[ $san_type == "address" ]]; then
		local san_flags="-fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls"
		if [[ -f "$ignorelist" ]]; then
			san_flags="$san_flags -fsanitize-blacklist=$ignorelist"
		fi

		# Build dependencies first with sanitizer flags
		echo "Building Redis dependencies with ASAN..."
		make -C deps \
			CC=clang \
			CXX=clang++ \
			CFLAGS="$san_flags" \
			CXXFLAGS="$san_flags" \
			hiredis linenoise lua fpconv fast_float \
			>> "$build_log" 2>&1 || true

		echo "Building Redis with CFLAGS: $san_flags"
		make -j$(nproc) \
			CC=clang \
			CXX=clang++ \
			OPTIMIZATION="-O1" \
			MALLOC="libc" \
			CFLAGS="$san_flags" \
			CXXFLAGS="$san_flags" \
			LDFLAGS="-fsanitize=address" \
			>> "$build_log" 2>&1 || build_result=$?
	fi

	if [[ $build_result -ne 0 ]]; then
		echo "Redis build failed. Last 50 lines of build log:"
		tail -50 "$build_log"
		cd "$ROOT"
		return 1
	fi

	if [[ -f "$redis_dir/src/redis-server" ]]; then
		REDIS_SERVER="$redis_dir/src/redis-server"
		echo "Redis built successfully: $REDIS_SERVER"
	else
		echo "Failed to build Redis with sanitizer - redis-server binary not found"
		echo "Last 50 lines of build log:"
		tail -50 "$build_log"
		cd "$ROOT"
		return 1
	fi

	cd "$ROOT"
}

setup_clang_sanitizer() {
	local ignorelist=$ROOT/tests/memcheck/redis.san-ignorelist
	if ! grep THPIsEnabled $ignorelist &> /dev/null; then
		echo "fun:THPIsEnabled" >> $ignorelist
	fi

	# for RediSearch module
	export RS_GLOBAL_DTORS=1

	# for RLTest
	export SANITIZER="$SAN"
	export SHORT_READ_BYTES_DELTA=512

	# --no-output-catch --exit-on-failure --check-exitcode
	RLTEST_SAN_ARGS="--sanitizer $SAN"

	if [[ $SAN == addr || $SAN == address ]]; then
		# Build Redis with ASAN to detect memory issues in Redis<->module interactions
		if [[ -z $REDIS_SERVER ]]; then
			build_redis_with_sanitizer "address" || {
				echo "Error: Failed to build Redis with ASAN."
				exit 1
			}
		fi

		export ASAN_OPTIONS="detect_odr_violation=0:halt_on_error=0:detect_leaks=1"
		export LSAN_OPTIONS="suppressions=$ROOT/tests/memcheck/asan.supp:use_tls=0"

	elif [[ $SAN == mem || $SAN == memory ]]; then
		# For MSAN, use regular redis-server (MSAN requires special libc build)
		if [[ -z $REDIS_SERVER ]]; then
			if command -v redis-server > /dev/null; then
				REDIS_SERVER=redis-server
			else
				echo "Error: No redis-server found. Please install redis-server."
				exit 1
			fi
		fi
	fi
}

#----------------------------------------------------------------------------------------------

setup_redis_server() {
	REDIS_SERVER=${REDIS_SERVER:-redis-server}

	if ! is_command $REDIS_SERVER; then
		echo "Cannot find $REDIS_SERVER. Aborting."
		exit 1
	fi
}

#----------------------------------------------------------------------------------------------

setup_valgrind() {
	# Use regular redis-server for valgrind tests
	if [[ -z $REDIS_SERVER ]]; then
		if command -v redis-server > /dev/null; then
			REDIS_SERVER=redis-server
		else
			echo "Error: No redis-server found. Please install redis-server."
			exit 1
		fi
	fi

	if [[ $VG_LEAKS == 0 ]]; then
		VG_LEAK_CHECK=no
		RLTEST_VG_NOLEAKS="--vg-no-leakcheck"
	else
		VG_LEAK_CHECK=full
		RLTEST_VG_NOLEAKS=""
	fi
	# RLTest reads this
	VG_OPTIONS="\
		-q \
		--leak-check=$VG_LEAK_CHECK \
		--show-reachable=no \
		--track-origins=yes \
		--show-possibly-lost=no"

	# To generate supressions and/or log to file
	# --gen-suppressions=all --log-file=valgrind.log

	VALGRIND_SUPRESSIONS=$ROOT/tests/memcheck/valgrind.supp

	RLTEST_VG_ARGS+="\
		--use-valgrind \
		--vg-verbose \
		$RLTEST_VG_NOLEAKS \
		--vg-no-fail-on-errors \
		--vg-suppressions $VALGRIND_SUPRESSIONS"


	# for module
	export RS_GLOBAL_DTORS=1

	# for RLTest
	export VALGRIND=1
	export VG_OPTIONS
	export RLTEST_VG_ARGS
}

#----------------------------------------------------------------------------------------------

setup_coverage() {
	export CODE_COVERAGE=1
}

#----------------------------------------------------------------------------------------------

run_env() {
	rltest_config=$(mktemp "${TMPDIR:-/tmp}/rltest.XXXXXXX")
	rm -f $rltest_config
	cat <<-EOF > $rltest_config
		--env-only
		--oss-redis-path=$REDIS_SERVER
		--module $MODULE
		--module-args '$MODARGS'
		$RLTEST_ARGS
		$RLTEST_TEST_ARGS
		$RLTEST_PARALLEL_ARG
		$RLTEST_VG_ARGS
		$RLTEST_SAN_ARGS
		$RLTEST_COV_ARGS

		EOF

	# Use configuration file in the current directory if it exists
	if [[ -n $CONFIG_FILE && -e $CONFIG_FILE ]]; then
		cat $CONFIG_FILE >> $rltest_config
	fi

	if [[ $VERBOSE == 1 || $NOP == 1 ]]; then
		echo "RLTest configuration:"
		cat $rltest_config
		[[ -n $VG_OPTIONS ]] && { echo "VG_OPTIONS: $VG_OPTIONS"; echo; }
	fi

	local E=0
	if [[ $NOP != 1 ]]; then
		{ $OP python3 -m RLTest @$rltest_config; (( E |= $? )); } || true
	else
		$OP python3 -m RLTest @$rltest_config
	fi

	[[ $KEEP != 1 ]] && rm -f $rltest_config

	return $E
}

#----------------------------------------------------------------------------------------------

run_tests() {
	local title="$1"
	shift
	if [[ -n $title ]]; then
		if [[ -n $GITHUB_ACTIONS ]]; then
			echo "::group::$title"
		else
			sep1
			printf "Running $title:\n\n"
		fi
	fi

	if [[ $EXT != 1 ]]; then
		rltest_config=$(mktemp "${TMPDIR:-/tmp}/rltest.XXXXXXX")
		rm -f $rltest_config
		if [[ $RLEC != 1 ]]; then
			cat <<-EOF > $rltest_config
				--oss-redis-path=$REDIS_SERVER
				--module $MODULE
				--module-args '$MODARGS'
				$RLTEST_ARGS
				$RLTEST_TEST_ARGS
				$RLTEST_PARALLEL_ARG
				$RLTEST_VG_ARGS
				$RLTEST_SAN_ARGS
				$RLTEST_COV_ARGS

				EOF
		else
			cat <<-EOF > $rltest_config
				$RLTEST_ARGS
				$RLTEST_TEST_ARGS
				$RLTEST_VG_ARGS

				EOF
		fi
	else # existing env
		if [[ $EXT == run ]]; then
			xredis_conf=$(mktemp "${TMPDIR:-/tmp}/xredis_conf.XXXXXXX")
			rm -f $xredis_conf
			cat <<-EOF > $xredis_conf
				loadmodule $MODULE $MODARGS
				EOF

			rltest_config=$(mktemp "${TMPDIR:-/tmp}/xredis_rltest.XXXXXXX")
			rm -f $rltest_config
			cat <<-EOF > $rltest_config
				--env existing-env
				$RLTEST_ARGS
				$RLTEST_TEST_ARGS

				EOF

			if [[ $VERBOSE == 1 ]]; then
				echo "External redis-server configuration:"
				cat $xredis_conf
			fi

			$REDIS_SERVER $xredis_conf &
			XREDIS_PID=$!
			echo "External redis-server pid: " $XREDIS_PID

		else # EXT=1
			rltest_config=$(mktemp "${TMPDIR:-/tmp}/xredis_rltest.XXXXXXX")
			[[ $KEEP != 1 ]] && rm -f $rltest_config
			cat <<-EOF > $rltest_config
				--env existing-env
				--existing-env-addr $EXT_HOST:$EXT_PORT
				$RLTEST_ARGS
				$RLTEST_TEST_ARGS

				EOF
		fi
	fi

	# Use configuration file in the current directory if it exists
	if [[ -n $CONFIG_FILE && -e $CONFIG_FILE ]]; then
		cat $CONFIG_FILE >> $rltest_config
	fi

	if [[ $VERBOSE == 1 || $NOP == 1 ]]; then
		echo "RLTest configuration:"
		cat $rltest_config
		[[ -n $VG_OPTIONS ]] && { echo "VG_OPTIONS: $VG_OPTIONS"; echo; }
	fi

	[[ $RLEC == 1 ]] && export RLEC_CLUSTER=1

	local E=0
	if [[ $NOP != 1 ]]; then
		{ $OP python3 -m RLTest @$rltest_config; (( E |= $? )); } || true
	else
		$OP python3 -m RLTest @$rltest_config
	fi

	[[ $KEEP != 1 ]] && rm -f $rltest_config

	if [[ -n $XREDIS_PID ]]; then
		echo "killing external redis-server: $XREDIS_PID"
		kill -TERM $XREDIS_PID
	fi

	if [[ -n $GITHUB_ACTIONS ]]; then
		echo "::endgroup::"
	fi
	return $E
}

#------------------------------------------------------------------------------------ Arguments

if [[ $1 == --help || $1 == help || $HELP == 1 ]]; then
	help
	exit 0
fi

OP=""
[[ $NOP == 1 ]] && OP=echo

[[ $V == 1 ]] && VERBOSE=1

#--------------------------------------------------------------------------------- Environments

DOCKER_HOST=${DOCKER_HOST:-127.0.0.1}
RLEC_PORT=${RLEC_PORT:-12000}

EXT_HOST=${EXT_HOST:-127.0.0.1}
EXT_PORT=${EXT_PORT:-6379}

PID=$$
OS=$(get_platform_os)
ARCH=$(get_platform_arch)
OSNICK=$(get_platform_osnick)

#---------------------------------------------------------------------------------- Tests scope

RLEC=${RLEC:-0}

if [[ $RLEC != 1 ]]; then
	GEN=${GEN:-1}
	AOF=${AOF:-1}
	TCK=${TCK:-1}
	UPGRADE=${UPGRADE:-1}

	if [[ -z $MODULE || ! -f $MODULE ]]; then
		echo "Module not found at ${MODULE}. Aborting."
		exit 1
	fi
else
	GEN=1
	AOF=0
	TCK=0
	UPGRADE=0
fi

#------------------------------------------------------------------------------------ Debugging

VG_LEAKS=${VG_LEAKS:-1}
VG_ACCESS=${VG_ACCESS:-1}

GDB=${GDB:-0}

if [[ $GDB == 1 ]]; then
	[[ $LOG != 1 ]] && RLTEST_LOG=0
	RLTEST_CONSOLE=1
fi

[[ $SAN == addr ]] && SAN=address
[[ $SAN == mem ]] && SAN=memory

if [[ -n $TEST ]]; then
	[[ $LOG != 1 ]] && RLTEST_LOG=0
	# export BB=${BB:-1}
	export RUST_BACKTRACE=1
fi

#---------------------------------------------------------------------------------- Parallelism

PARALLEL=${PARALLEL:-1}

[[ $EXT == 1 || $EXT == run || $BB == 1 || $GDB == 1 || -n $TEST ]] && PARALLEL=0

if [[ -n $PARALLEL && $PARALLEL != 0 ]]; then
	if [[ $PARALLEL == 1 ]]; then
		parallel="$(get_nproc)"
	else
		parallel="$PARALLEL"
	fi
	RLTEST_PARALLEL_ARG="--parallelism $parallel"
fi

#------------------------------------------------------------------------------- Test selection

if [[ -n $TEST ]]; then
	RLTEST_TEST_ARGS+=$(echo -n " "; echo "$TEST" | awk 'BEGIN { RS=" "; ORS=" " } { print "--test " $1 }')
fi

if [[ -n $TESTFILE ]]; then
	if ! is_abspath "$TESTFILE"; then
		TESTFILE="$ROOT/$TESTFILE"
	fi
	RLTEST_TEST_ARGS+=" -f $TESTFILE"
fi

if [[ -n $FAILEDFILE ]]; then
	if ! is_abspath "$FAILEDFILE"; then
		TESTFILE="$ROOT/$FAILEDFILE"
	fi
	RLTEST_TEST_ARGS+=" -F $FAILEDFILE"
fi

if [[ $LIST == 1 ]]; then
	NO_SUMMARY=1
	RLTEST_ARGS+=" --collect-only"
fi

#---------------------------------------------------------------------------------------- Setup

if [[ $VERBOSE == 1 ]]; then
	RLTEST_VERBOSE=1
fi

RLTEST_LOG=${RLTEST_LOG:-$LOG}

if [[ $COV == 1 ]]; then
	setup_coverage
fi

if [[ -n $REDIS_PORT ]]; then
	RLTEST_ARGS+="--redis-port $REDIS_PORT"
fi

[[ $UNIX == 1 ]] && RLTEST_ARGS+=" --unix"
[[ $RANDPORTS == 1 ]] && RLTEST_ARGS+=" --randomize-ports"

#----------------------------------------------------------------------------------------------

setup_rltest

if [[ -n $SAN ]]; then
	setup_clang_sanitizer
	RLTEST_ARGS+=" --test-timeout 900"
elif [[ $VG == 1 ]]; then
	# no timeout for Valgrind tests
	setup_valgrind
else
	RLTEST_ARGS+=" --test-timeout 180"
fi

if [[ $RLEC != 1 ]]; then
	setup_redis_server
fi

#------------------------------------------------------------------------------------- Env only

if [[ $ENV_ONLY == 1 ]]; then
	run_env
	exit 0
fi

#-------------------------------------------------------------------------------- Running tests

if [[ $CLEAR_LOGS != 0 ]]; then
	rm -rf $HERE/logs $HERE/../tck/logs $HERE/../upgrade/logs
fi

if [[ $OS == macos ]]; then
	runn ulimit -n 10000
fi

E=0

if [[ $GEN == 1 ]]; then
	{ (run_tests "general tests"); (( E |= $? )); } || true
fi

if [[ $AOF == 1 ]]; then
	if [[ -z $TEST || $TEST == test_persistency* ]]; then
		{ (RLTEST_ARGS="${RLTEST_ARGS} --use-aof" RLTEST_TEST_ARGS="--test test_persistency" \
		   run_tests "tests with AOF"); (( E |= $? )); } || true
	else
		AOF=0
	fi
fi

if [[ $TCK == 1 ]]; then
	if [[ -z $TEST ]]; then
		{ (cd $HERE/../tck; run_tests "TCK tests"); (( E |= $? )); } || true
	else
		TCK=0
	fi
fi

if [[ $UPGRADE == 1 ]]; then
	if [[ -z $TEST ]]; then
		{ (cd $HERE/../upgrade; run_tests "Upgrade tests"); (( E |= $? )); } || true
	else
		UPGRADE=0
	fi
fi

if [[ $RLEC == 1 ]]; then
	dhost=$(echo "$DOCKER_HOST" | awk -F[/:] '{print $4}')
	{ (RLTEST_ARGS="${RLTEST_ARGS} --env existing-env --existing-env-addr $dhost:$RLEC_PORT" \
	   run_tests "tests on RLEC"); (( E |= $? )); } || true
fi

#-------------------------------------------------------------------------------------- Summary

if [[ $NO_SUMMARY == 1 ]]; then
	exit 0
fi

if [[ $NOP != 1 ]]; then
	if [[ -n $SAN || $VG == 1 ]]; then
		# Build list of test directories to check
		MEMCHECK_DIRS=()
		[[ $GEN == 1 || $AOF == 1 ]] && MEMCHECK_DIRS+=(flow)
		[[ $TCK == 1 ]] && MEMCHECK_DIRS+=(tck)

		{ memcheck_summary "${MEMCHECK_DIRS[@]}"; (( E |= $? )); } || true
	fi
fi

exit $E
