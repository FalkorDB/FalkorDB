// Calibration for the ri_instructions offset used by tensor_bench.c:
// a loop whose per-iteration instruction count is known from its disassembly.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <libproc.h>
#include <unistd.h>

static uint64_t ins(void) {
	static uint8_t b[1024];
	proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)b);
	uint64_t v;
	memcpy(&v, b + 16 + 29 * 8, 8);
	return v;
}

volatile uint64_t sink;

int main(void) {
	for(int rep = 0; rep < 3; rep++) {
		uint64_t n = 1000000000ULL, a = 0;
		uint64_t i0 = ins();
		for(uint64_t i = 0; i < n; i++) a += i;
		uint64_t i1 = ins();
		sink = a;
		printf("rep%d: %.4f instr/iter (total %llu)\n", rep,
		       (double)(i1 - i0) / n, (unsigned long long)(i1 - i0));
	}
	return 0;
}
