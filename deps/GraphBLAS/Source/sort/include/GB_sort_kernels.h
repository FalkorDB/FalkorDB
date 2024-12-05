//------------------------------------------------------------------------------
// GB_sort_kernels.h: definitions for sorting functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SORT_KERNELS_H
#define GB_SORT_KERNELS_H

//------------------------------------------------------------------------------
// GB_lt_1: sorting comparator function, one key
//------------------------------------------------------------------------------

// A [a] and B [b] are keys of one integer.

// GB_lt_1 returns true if A [a] < B [b], for GB_qsort_1b

#define GB_lt_1(A_0, a, B_0, b) (A_0 [a] < B_0 [b])

//------------------------------------------------------------------------------
// GB_lt_2: sorting comparator function, two keys
//------------------------------------------------------------------------------

// A [a] and B [b] are keys of two integers.

// GB_lt_2 returns true if A [a] < B [b], for GB_qsort_2 and GB_msort_2

#define GB_lt_2(A_0, A_1, a, B_0, B_1, b)                                   \
(                                                                           \
    (A_0 [a] < B_0 [b]) ?                                                   \
    (                                                                       \
        true                                                                \
    )                                                                       \
    :                                                                       \
    (                                                                       \
        (A_0 [a] == B_0 [b]) ?                                              \
        (                                                                   \
            /* primary key is the same; tie-break on the 2nd key */         \
            (A_1 [a] < B_1 [b])                                             \
        )                                                                   \
        :                                                                   \
        (                                                                   \
            false                                                           \
        )                                                                   \
    )                                                                       \
)

//------------------------------------------------------------------------------
// GB_lt_3: sorting comparator function, three keys
//------------------------------------------------------------------------------

// A [a] and B [b] are keys of three integers.

// GB_lt_3 returns true if A [a] < B [b], for GB_qsort_3 and GB_msort_3

#define GB_lt_3(A_0, A_1, A_2, a, B_0, B_1, B_2, b)                         \
(                                                                           \
    (A_0 [a] < B_0 [b]) ?                                                   \
    (                                                                       \
        true                                                                \
    )                                                                       \
    :                                                                       \
    (                                                                       \
        (A_0 [a] == B_0 [b]) ?                                              \
        (                                                                   \
            /* primary key is the same; tie-break on the 2nd and 3rd key */ \
            GB_lt_2 (A_1, A_2, a, B_1, B_2, b)                              \
        )                                                                   \
        :                                                                   \
        (                                                                   \
            false                                                           \
        )                                                                   \
    )                                                                       \
)

//------------------------------------------------------------------------------
// GB_eq_*: sorting comparator function, one to three keys
//------------------------------------------------------------------------------

// A [a] and B [b] are keys of two or three integers.
// GB_eq_* returns true if A [a] == B [b]

#define GB_eq_3(A_0, A_1, A_2, a, B_0, B_1, B_2, b)                         \
(                                                                           \
    (A_0 [a] == B_0 [b]) &&                                                 \
    (A_1 [a] == B_1 [b]) &&                                                 \
    (A_2 [a] == B_2 [b])                                                    \
)

#define GB_eq_2(A_0, A_1, a, B_0, B_1, b)                                   \
(                                                                           \
    (A_0 [a] == B_0 [b]) &&                                                 \
    (A_1 [a] == B_1 [b])                                                    \
)

#define GB_eq_1(A_0, a, B_0, b)                                             \
(                                                                           \
    (A_0 [a] == B_0 [b])                                                    \
)

#endif

