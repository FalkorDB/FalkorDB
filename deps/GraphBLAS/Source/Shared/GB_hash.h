//------------------------------------------------------------------------------
// GB_hash.h: definitions for hashing
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_HASH_H
#define GB_HASH_H

//------------------------------------------------------------------------------
// functions for the Hash method for C=A*B, and for the A->Y hyperhash
//------------------------------------------------------------------------------

// initial hash function, for where to place the integer i in the hash table.
// hash_bits is a bit mask to compute the result modulo the hash table size,
// which is always a power of 2.  The function is (i*257) & (hash_bits).
#define GB_HASHF(i,hash_bits) ((((i) << 8) + (i)) & (hash_bits))

// hash function for the A->Y hyperhash:
// lots of intentional collisions: but blocks are scattered
#define GB_HASHF2(i,hash_bits) ((((i) >> 2) + 17L*((i) >> 8)) & (hash_bits))

// rehash function, for subsequent hash lookups if the initial hash function
// refers to a hash entry that is already occupied.  Linear probing is used,
// so the function does not currently depend on i.  On input, hash is equal
// to the current value of the hash function, and on output, hash is set to
// the new hash value.
#define GB_REHASH(hash,i,hash_bits) hash = ((hash + 1) & (hash_bits))

// The hash functions and their parameters are modified from this paper:

// [2] Yusuke Nagasaka, Satoshi Matsuoka, Ariful Azad, and Aydin Buluc. 2018.
// High-Performance Sparse Matrix-Matrix Products on Intel KNL and Multicore
// Architectures. In Proc. 47th Intl. Conf. on Parallel Processing (ICPP '18).
// Association for Computing Machinery, New York, NY, USA, Article 34, 1–10.
// DOI:https://doi.org/10.1145/3229710.3229720

// The hash function in that paper is (i*107)&(hash_bits).  Here, the term
// 107 is replaced with 257 to allow for a faster hash function computation.

#endif

