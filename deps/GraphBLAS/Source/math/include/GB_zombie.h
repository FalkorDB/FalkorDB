//------------------------------------------------------------------------------
// GB_zombie.h: definitions for zombies
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ZOMBIE_H
#define GB_ZOMBIE_H

// An entry A(i,j) in a matrix can be marked as a "zombie".  A zombie is an
// entry that has been marked for deletion, but hasn't been deleted yet because
// it's more efficient to delete all zombies all at once, instead of one at a
// time.  Zombies are created by submatrix assignment, C(I,J)=A which copies
// not only new entries into C, but it also deletes entries already present in
// C.  If an entry appears in A but not C(I,J), it is a new entry; new entries
// placed in the pending tuple lists to be added later.  If an entry appear in
// C(I,J) but NOT in A, then it is marked for deletion by marking its row index
// as a zombie.

// Zombies can be restored as regular entries by GrB_*assign.  If an assignment
// C(I,J)=A finds an entry in A that is a zombie in C, the zombie becomes a
// regular entry, taking on the value from A.  The row index is 'dezombied'.

// Zombies are deleted and pending tuples are added into the matrix all at
// once, by GB_wait.

// For GraphBLAS 10.0.0 and later, the zombie function has changed to allow for
// a larger range of valid indices when using 32-bit integers, where now
// GB_ZOMBIE([0 1 2 3 ... INT32_MAX]) = [-1 -2 -3 ... INT32_MIN].  The zombie
// function is zombie (i) = -i-1 or simply ~i, the one's complement of i.  This
// allows the largest index of a 32-bit A->i array to be INT32_MAX, giving a
// maximum matrix dimension of exactly 2^31 when 32-bit indices are used.

// With this change, there is no neutral element x for which zombie (x) == x,
// but this feature is not required for the

// Some algorithms need more space than this for their indices, at least
// temporarily.  GrB_mxm (saxpy3) on the CPU uses a 4-state finite state
// machine held in the Hf array (not in C->i itself), but this Hf array can
// remain int64_t even when C->i is 32-bit.  GrB_mxm (dot3) on the GPU requires
// 4 bits for its buckets; for 32-bit matrices, the bucket assignments need to
// be stored in a separate array; they won't fit in C->i unless the matrix
// dimension is about 2^28 or smaller.

// The max matrix dimensions for 64-bit integer matrices could be increased to
// to about 2^62 on the CPU.  This would still be OK for the Hf [hash] entries
// for the fine Hash method.  The GPU currently is using 4 bits for up to 16
// buckets ... but it is currently only using about 4 buckets, requiring just
// two bits for the bucket index.

// For the zombie detection to work in the GB_IS_ZOMBIE and GB_UNZOMBIE
// functions, the integer i must be a signed integer, either int32_t or
// int64_t.  These functions could be revised to work with unsigned integers,
// but they would differ for 32-bit and 64-bit integers.  In addition, the
// typecast of { uint64_t i = Ai [p] ; i = GB_UNZOMBIE (i) ; } would fail if Ai
// is uint32_t since typecasting of unsigned integers does have a sign bit to
// extend.  Thus, when zombies are present, the Ai array must be treated as
// int32_t or int64_t, and the temporary index i can then always be int64_t.

// Thus, if Ai is 32-bit, use the following:
//
//      int32_t *Ai = A->i ;
//      ...
//      int64_t i = Ai [p] ;    // extends the sign bit if Ai [p] is a zombie
//      i = GB_UNZOMBIE (i) ;   // converts i to nonzombie
//
// If Ai is 64-bit, use the following:
//
//      int64_t *Ai = A->i ;
//      ...
//      int64_t i = Ai [p] ;    // extends the sign bit if Ai [p] is a zombie
//      i = GB_UNZOMBIE (i) ;   // converts i to nonzombie

// Note in the above two snippets of code, the 2nd two statemnts are identical.
// This facilitates the construction of code that handles both 32-bit and
// 64-bit integers for A->i.  Thus, for handling 32/64 bit integers for A->i:
//
//      GB_Ai_DECLARE (Ai, ) ;      // void *Ai ; int32_t *Ai32 ; int64_t *Ai64
//      GB_Ai_PTR (Ai, A) ;         // Ai32 = A->i or Ai64 = A->i
//      ...
//      int64_t i = GB_IGET (Ai, p) ;   // i = Ai32 [p] or Ai64 [p]
//      i = GB_UNZOMBIE (i) ;       // converts i to nonzombie

// In a JIT kernel, when Ai is 64-bit the above code becomes the following,
// after specialization of all the macros:
//
//      int64_t *restrict Ai = NULL ;
//      Ai = (A) ? A->i : NULL ;
//      ...
//      int64_t i = Ai [p] ;
//      i = (i < 0) ? (~i) : i ;
//
// when Ai is 32-bit, the JIT kernel code becomes:
//
//      int32_t *restrict Ai = NULL ;
//      Ai = (A) ? A->i : NULL ;
//      ...
//      int64_t i = Ai [p] ;            // note the typecast
//      i = (i < 0) ? (~i) : i ;

// Outside of a JIT kernel, the code is a little more complex, becoming:
//
//      void *Ai = NULL ;
//      int32_t *Ai32 = NULL ;
//      int64_t *Ai64 = NULL ;
//      Ai = (A) ? A->i : NULL ;
//      Ai32 = (A) ? (A->i_is_32 ? Ai : NULL) : NULL ;
//      Ai64 = (A) ? (A->i_is_32 ? NULL : Ai) : NULL ;
//      ...
//      int64_t i = (Ai32 ? Ai32 [k] : Ai64 [k]) ;
//      i = (i < 0) ? (~i) : i ;

// In the above examples, the "..." separates the pointer initialization, which
// happens just once, and the access of each entry, which can happen many
// times.  In code outside of a JIT kernel, the hardware branch predictor will
// help with the Ai32 ternary expression, since Ai32 typically does not change.
// It seems to be fast enough in practice.

// To ensure the ternary expression appears just once, to help clarify the
// effect of the typecast from the int32_t array Ai to int64_t scalar i, and to
// ensure the correct 32/64 bit pointer is used for Ai, all zombie functions
// should be applied only to temporary scalars, as i = GB_UNZOMBIE (i) or
// GB_IS_ZOMBIE (i), not i = GB_UNZOMBIE (Ai [p]) or GB_IS_ZOMBIE (Ai [p]).

#define GB_ZOMBIE(i)        (~(i))
#define GB_DEZOMBIE(i)      GB_ZOMBIE (i)
#define GB_IS_ZOMBIE(i)     ((i) < 0)
#define GB_UNZOMBIE(i)      (GB_IS_ZOMBIE (i) ? GB_DEZOMBIE (i) : (i))

// Note that GB_ZOMBIE and GB_DEZOMBIE are identical.  GB_ZOMBIE (i) is used
// when i is known to not be a zombie, and the result of the function is the
// zombie index for i.  GB_DEZOMBIE is used when i is known to be a zombified
// index, and the result is the non-zombie index for that entry.  The existence
// of the two function names is only for code clarity.

// GB_UNZOMBIE (i) is used when the index i may or may not be a zombie, and
// the result of the function is the non-zombie index for i.

#endif

