//------------------------------------------------------------------------------
// GB_zombie.h: definitions for zombies
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ZOMBIE_H
#define GB_ZOMBIE_H

// GB_ZOMBIE is a kind of "negation" about (-1) of a zero-based index.
// If i >= 0 then it is not a zombie.
// If i < 0 then it has been marked as a zombie.
// Like negation, GB_ZOMBIE is its own inverse: GB_ZOMBIE (GB_ZOMBIE (i)) == i.
// The "nil" value, -1, doesn't change: GB_ZOMBIE (-1) = -1.
// GB_UNZOMBIE(i) is like taking an absolute value, undoing any GB_ZOMBIE(i).

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
// regular entry, taking on the value from A.  The row index is 'dezombied'

// Zombies are deleted and pending tuples are added into the matrix all at
// once, by GB_wait.

/* OLD:
#define GB_FLIP(i)             (-(i)-2)
#define GB_IS_FLIPPED(i)       ((i) < 0)
#define GB_IS_ZOMBIE(i)        ((i) < 0)
#define GB_IS_NOT_FLIPPED(i)   ((i) >= 0)
#define GB_UNFLIP(i)           (((i) < 0) ? GB_FLIP(i) : (i))
#define GBI_UNFLIP(Ai,p,avlen)      \
    ((Ai == NULL) ? ((p) % (avlen)) : GB_UNFLIP (Ai [p]))
*/

// NEW:
//  replace GB_FLIP with GB_ZOMBIE.
//  add GB_DEZOMBIE (same as GB_FLIP and GB_ZOMBIE).
//  no change to GB_IS_ZOMBIE.
//  delete GB_IS_FLIPPED and GB_IS_NOT_FLIPPED.
//  replace GB_UNFLIP with GB_UNZOMBIE.
//  replace GBI_UNFLIP with GBI_UNZOMBIE.

#define GB_ZOMBIE(i)        (-(i)-2)
#define GB_DEZOMBIE(i)      (-(i)-2)

#define GB_IS_ZOMBIE(i)     ((i) < 0)

#define GB_UNZOMBIE(i)      (((i) < 0) ? GB_ZOMBIE(i) : (i))

#define GBI_UNZOMBIE(Ai,p,avlen)      \
    ((Ai == NULL) ? ((p) % (avlen)) : GB_UNZOMBIE (Ai [p]))

#endif

