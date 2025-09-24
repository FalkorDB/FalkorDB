/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// this is a modified version of Isaac Turner 29 April 2014 Public Domain
// sort_r.h
// https://github.com/noporpoise/sort_r/blob/master/sort_r.h

#pragma once

#include <stdlib.h>

#if (defined __APPLE__ || defined __MACH__ || defined __DARWIN__ || defined BSD)
# define _SORT_R_BSD
#elif (defined _GNU_SOURCE || defined __GNU__ || defined __linux__ || \
		defined __GLIBC__)
# define _SORT_R_LINUX
#endif

#if defined _SORT_R_BSD
// BSD (qsort_r) require argument swap

struct sort_r_data {
	void *arg;
	int (*compar)(const void *_a, const void *_b, void *_arg);
};

static int sort_r_arg_swap
(
	void *s,
	const void *a,
	const void *b
) {
	struct sort_r_data *ss = (struct sort_r_data*)s;
	return (ss->compar)(a, b, ss->arg);
}

#endif

static inline void sort_r
(
	void *base,
	size_t nel,
	size_t width,
	int (*compar)(const void *_a, const void *_b, void *_arg),
	void *arg
) {
	#if defined _SORT_R_LINUX
		qsort_r(base, nel, width, compar, arg);
	#elif defined _SORT_R_BSD
		struct sort_r_data tmp;
		tmp.arg = arg;
		tmp.compar = compar;
		qsort_r(base, nel, width, &tmp, sort_r_arg_swap);
	#endif
}

#undef _SORT_R_LINUX
#undef _SORT_R_BSD

