
#pragma once

#if (defined(DEBUG) || defined(_DEBUG)) && !defined(NDEBUG)

#include <stdbool.h>

extern bool __via_gdb;

#if defined(__i386__) || defined(__x86_64__)
#	define _BB do { if (__via_gdb) { __asm__("int $3"); } } while(0)

#elif defined(__arm__)
#	define _BB do { if (__via_gdb) { __asm__("trap"); } } while(0)

#elif defined(__aarch64__)
#	if defined(__APPLE__)
#		define _BB do { if (__via_gdb) { __builtin_debugtrap(); } } while(0)
#	else
#		define _BB do { if (__via_gdb) { __asm__(".inst 0xd4200000"); } } while(0)
#	endif

#else
#	include <signal.h>
#	define _BB do { if (__via_gdb) { raise(SIGTRAP); } } while(0)
#endif

#elif defined(READIES_ALLOW_BB)

#define _BB do {} while(0)

#endif // DEBUG
