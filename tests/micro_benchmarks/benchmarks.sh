#!/bin/bash

PROGNAME="${BASH_SOURCE[0]}"
HERE="$(cd "$(dirname "$PROGNAME")" &>/dev/null && pwd)"
ROOT=$(cd $HERE/../.. && pwd)

cd $HERE

#----------------------------------------------------------------------------------------------

help() {
    cat <<-'END'
        Run micro-benchmarks

        [ARGVARS...] micro-benchmarks [--help|help]

        Argument variables:
        BINROOT=path   Path to repo binary root dir
        BENCH=name     Operate in single-benchmark mode
        JSON=1         Output results in JSON format to results directory

        SAN=addr|mem   Run with sanitizer
        VG=1           Run with Valgrind
        LEAK=1         Run benchmark that leaks (for sanitizer diagnostics)

        GDB=1          Enable interactive gdb debugging (in single-benchmark mode)
        CLANG=1        Implies use of lldb debugger
        VERBOSE=1      Print commands and Redis output
        NOP=1          Dry run
        HELP=1         Show help

END
}

#----------------------------------------------------------------------------------------------

sanitizer_defs() {
    if [[ -n $SAN ]]; then
        ASAN_LOG=${LOGS_DIR}/${BENCH_NAME}.asan.log
        export ASAN_OPTIONS="detect_odr_violation=0:halt_on_error=0::detect_leaks=1:log_path=${ASAN_LOG}"
        export LSAN_OPTIONS="suppressions=$ROOT/tests/memcheck/asan.supp:use_tls=0"
    fi
}

#----------------------------------------------------------------------------------------------

sanitizer_summary() {
	if grep -l "leaked in" ${LOGS_DIR}/*.asan.log* &> /dev/null; then
		echo
		echo "${LIGHTRED}Sanitizer: leaks detected:${RED}"
		grep -l "leaked in" ${LOGS_DIR}/*.asan.log*
		echo "${NOCOLOR}"
		E=1
	fi
	if grep -l "dynamic-stack-buffer-overflow" ${LOGS_DIR}/*.asan.log* &> /dev/null; then
		echo
		echo "${LIGHTRED}Sanitizer: buffer overflow detected:${RED}"
		grep -l "dynamic-stack-buffer-overflow" ${LOGS_DIR}/*.asan.log*
		echo "${NOCOLOR}"
		E=1
	fi
	if grep -l "stack-buffer-overflow" ${LOGS_DIR}/*.asan.log* &> /dev/null; then
		echo
		echo "${LIGHTRED}Sanitizer: buffer overflow detected:${RED}"
		grep -l "stack-buffer-overflow" ${LOGS_DIR}/*.asan.log*
		echo "${NOCOLOR}"
		E=1
	fi
	if grep -l "stack-use-after-scope" ${LOGS_DIR}/*.asan.log* &> /dev/null; then
		echo
		echo "${LIGHTRED}Sanitizer: stack use after scope detected:${RED}"
		grep -l "stack-use-after-scope" ${LOGS_DIR}/*.asan.log*
		echo "${NOCOLOR}"
		E=1
	fi
}

#----------------------------------------------------------------------------------------------

[[ $1 == --help || $1 == help || $HELP == 1 ]] && { help; exit 0; }

OP=
[[ $NOP == 1 ]] && OP=echo

LEAK=${LEAK:-0}
JSON=${JSON:-0}

export LOGS_DIR=$ROOT/tests/micro_benchmarks/logs
export RESULTS_DIR=$ROOT/tests/micro_benchmarks/results

# Create results directory if JSON output is requested
if [[ $JSON == 1 ]]; then
    mkdir -p $RESULTS_DIR
fi

if [[ $GDB == 1 ]]; then
    if [[ $CLANG == 1 ]]; then
        GDB_CMD="lldb -o run --"
    else
        GDB_CMD="gdb -ex r --args"
    fi
else
    GDB_CMD=
fi

VG_OP=
if [[ $VG == 1 ]]; then
    VG_OP=valgrind
    VG_SUPRESSIONS=$ROOT/tests/micro_benchmarks/mircobench.supp
    VG_OPTIONS="\
        --error-exitcode=0 \
        --leak-check=full \
        --track-origins=yes \
        --suppressions=${VG_SUPRESSIONS}"
    if [[ $FULL == 1 ]]; then
        VG_OPTIONS+=" \
            --show-reachable=yes \
            --show-possibly-lost=yes"
    else
        VG_OPTIONS+=" \
            --show-reachable=no \
            --show-possibly-lost=no"
    fi
fi

#----------------------------------------------------------------------------------------------

if [[ $CLEAR_LOGS != 0 ]]; then
    rm -rf $LOGS_DIR
fi
mkdir -p $LOGS_DIR

if [[ -z $BINROOT || ! -d $BINROOT ]]; then
    eprint "BINROOT not defined or nonexistant"
    exit 1
fi

if [[ $LIST == 1 ]]; then
    BENCH_ARGS+=" --list"
fi

E=0

echo "========================================"
echo "# Running micro benchmarks"
BENCHS_DIR="$(cd $BINROOT/tests/micro_benchmarks 2>/dev/null && pwd)"

if [[ ! -d "$BENCHS_DIR" ]]; then
    echo "Error: Benchmark directory not found: $BINROOT/tests/micro_benchmarks"
    exit 1
fi

cd $ROOT/tests/micro_benchmarks
if [[ -z $BENCH ]]; then
    for bench in $(find $BENCHS_DIR -type f -executable -name "benchmark_*"); do
        if [[ ! -x $bench ]]; then
            continue
        fi
        bench_name="$(basename $bench)"
        if [[ $LEAK == 1 || $bench_name != benchmark_leak ]]; then
            echo "Running $bench ..."
            
            # Prepare benchmark arguments
            BENCH_ARGS=""
            if [[ $JSON == 1 ]]; then
                BENCH_ARGS="--benchmark_format=json --benchmark_out=${RESULTS_DIR}/${bench_name}_results.json"
            fi
            
            if [[ $VG == 1 ]]; then
                VG_LOG_ARG="--log-file=${LOGS_DIR}/${bench_name}.valgrind.log"
                { $OP $VG_OP $VG_OPTIONS $VG_LOG_ARG $bench $BENCH_ARGS; (( E |= $? )); } || true
            else
                BENCH_NAME="$bench_name" sanitizer_defs
                { $OP $bench $BENCH_ARGS; (( E |= $? )); } || true
            fi
        fi
    done
else
    SUPERBENCH=$(echo "$BENCH" | cut -d: -f1)
    SUBBENCH=$(echo "$BENCH" | cut -s -d: -f2)
    echo SUPERBENCH=$SUPERBENCH
    echo SUBBENCH=$SUBBENCH
    echo "Running $BENCHS_DIR/$SUPERBENCH ..."
    
    # Prepare benchmark arguments
    BENCH_ARGS="$SUBBENCH"
    if [[ $JSON == 1 ]]; then
        BENCH_ARGS="--benchmark_format=json --benchmark_out=${RESULTS_DIR}/${SUPERBENCH}_results.json $SUBBENCH"
    fi
    
    if [[ $VG == 1 ]]; then
        VG_LOG_ARG="--log-file=${LOGS_DIR}/${SUPERBENCH}.valgrind.log"
        { $OP $VG_OP $VG_OPTIONS $VG_LOG_ARG $BENCHS_DIR/$SUPERBENCH $BENCH_ARGS; (( E |= $? )); } || true
    else
        BENCH_NAME="$SUPERBENCH" sanitizer_defs
        { $OP $GDB_CMD $BENCHS_DIR/$SUPERBENCH $BENCH_ARGS; (( E |= $? )); } || true
    fi
fi

if [[ -n $SAN || $VG == 1 ]]; then
    # sanitizer_summary
    { UNIT=1 $ROOT/sbin/memcheck-summary.sh; (( E |= $? )); } || true
fi

exit $E
