#!/bin/bash

PROGNAME="${BASH_SOURCE[0]}"
HERE="$(cd "$(dirname "$PROGNAME")" &>/dev/null && pwd)"
ROOT=$(cd $HERE/../.. && pwd)

# Source common definitions and functions
. "$ROOT/tests/common.sh"

cd $HERE

#----------------------------------------------------------------------------------------------

help() {
	cat <<-'END'
		Run unit tests

		[ARGVARS...] unit-tests [--help|help]

		Argument variables:
		BINROOT=path   Path to repo binary root dir
		TEST=name      Operate in single-test mode

		SAN=addr|mem   Run with sanitizer
		VG=1           Run with Valgrind
		LEAK=1         Run test that leaks (for sanitizer diagnostics)

		GDB=1          Enable interactive gdb debugging (in single-test mode)
		CLANG=1        Implies use of lldb debugger
		VERBOSE=1      Print commands and Redis output
		NOP=1          Dry run
		HELP=1         Show help


	END
}

#----------------------------------------------------------------------------------------------

sanitizer_defs() {
	if [[ -n $SAN ]]; then
		ASAN_LOG=${LOGS_DIR}/${TEST_NAME}.asan.log
		export ASAN_OPTIONS="detect_odr_violation=0:halt_on_error=0::detect_leaks=1:log_path=${ASAN_LOG}"
		export LSAN_OPTIONS="suppressions=$ROOT/tests/memcheck/asan.supp:use_tls=0"
	fi
}

#----------------------------------------------------------------------------------------------

sanitizer_summary() {
	if grep -l "leaked in" ${LOGS_DIR}/*.asan.log* &> /dev/null; then
		echo
		echo -e "${LIGHTRED}Sanitizer: leaks detected:${RED}"
		grep -l "leaked in" ${LOGS_DIR}/*.asan.log*
		echo -e "${NOCOLOR}"
		E=1
	fi
	if grep -l "dynamic-stack-buffer-overflow" ${LOGS_DIR}/*.asan.log* &> /dev/null; then
		echo
		echo -e "${LIGHTRED}Sanitizer: buffer overflow detected:${RED}"
		grep -l "dynamic-stack-buffer-overflow" ${LOGS_DIR}/*.asan.log*
		echo -e "${NOCOLOR}"
		E=1
	fi
	if grep -l "stack-buffer-overflow" ${LOGS_DIR}/*.asan.log* &> /dev/null; then
		echo
		echo -e "${LIGHTRED}Sanitizer: buffer overflow detected:${RED}"
		grep -l "stack-buffer-overflow" ${LOGS_DIR}/*.asan.log*
		echo -e "${NOCOLOR}"
		E=1
	fi
	if grep -l "stack-use-after-scope" ${LOGS_DIR}/*.asan.log* &> /dev/null; then
		echo
		echo -e "${LIGHTRED}Sanitizer: stack use after scope detected:${RED}"
		grep -l "stack-use-after-scope" ${LOGS_DIR}/*.asan.log*
		echo -e "${NOCOLOR}"
		E=1
	fi
}

#----------------------------------------------------------------------------------------------

[[ $1 == --help || $1 == help || $HELP == 1 ]] && { help; exit 0; }

OP=
[[ $NOP == 1 ]] && OP=echo

LEAK=${LEAK:-0}

export LOGS_DIR=$ROOT/tests/unit/logs

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
	# VG_SUPRESSIONS=$ROOT/tests/memcheck/valgrind.supp
	VG_SUPRESSIONS=$ROOT/tests/unit/unittests.supp
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
	TEST_ARGS+=" --list"
fi

E=0

separator
echo "# Running unit tests"
TESTS_DIR="$(cd $BINROOT/tests/unit; pwd)"
cd $ROOT/tests/unit
if [[ -z $TEST ]]; then
	for test in $(find $TESTS_DIR -name "test_*" -type f -print); do
		if [[ ! -x $test ]]; then
			continue
		fi
		test_name="$(basename $test)"
		if [[ $LEAK == 1 || $test_name != test_leak ]]; then
			echo "Running $test ..."
			if [[ $VG == 1 ]]; then
				VG_LOG_ARG="--log-file=${LOGS_DIR}/${test_name}.valgrind.log"
				{ $OP $VG_OP $VG_OPTIONS $VG_LOG_ARG $test; (( E |= $? )); } || true
			else
				TEST_NAME="$test_name" sanitizer_defs
				{ $OP $test $TEST_ARGS; (( E |= $? )); } || true
			fi
		fi
	done
else
	SUPERTEST=$(echo "$TEST" | cut -d: -f1)
	SUBTEST=$(echo "$TEST" | cut -s -d: -f2)
	echo SUPERTEST=$SUPERTEST
	echo SUBTEST=$SUBTEST
	echo "Running $TESTS_DIR/$SUPERTEST ..."
	if [[ $VG == 1 ]]; then
		VG_LOG_ARG="--log-file=${LOGS_DIR}/${SUPERTEST}.valgrind.log"
		{ $OP $VG_OP $VG_OPTIONS $VG_LOG_ARG $TESTS_DIR/$SUPERTEST $TEST_ARGS $SUBTEST; (( E |= $? )); } || true
	else
		TEST_NAME="$SUPERTEST" sanitizer_defs
		{ $OP $GDB_CMD $TESTS_DIR/$SUPERTEST $TEST_ARGS $SUBTEST; (( E |= $? )); } || true
	fi
fi

if [[ -n $SAN || $VG == 1 ]]; then
	{ memcheck_summary unit; (( E |= $? )); } || true
fi

exit $E
