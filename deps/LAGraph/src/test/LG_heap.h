//------------------------------------------------------------------------------
// LG_heap: a Heap data structure and its operations
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// The Heap is an array of LG_Elements: Heap [1..nheap].  Each entry in the
// Heap is a LG_Element, with a key and name.  LG_Element must be defined
// by the including file.  For example:

/*
    typedef int64_t LG_key_t ;
    typedef struct
    {
        int64_t name ;
        LG_key_t key ;
    }
    LG_Element ;
    #include "LG_heap.h"
*/

#ifndef LG_HEAP_H
#define LG_HEAP_H

#undef  LG_FREE_ALL
#define LG_FREE_ALL ;

// These methods assume the caller allocates all memory, so no brutal memory
// test is needed.

//------------------------------------------------------------------------------
// LG_iheap_check: make sure Iheap is correct
//------------------------------------------------------------------------------

// Ensure that e == Heap [p] implies p == Iheap [e.name] for all entries
// in the heap.  Also ensure that e.name is in the range 0:n-1.

static inline int LG_iheap_check
(
    // input:
    const LG_Element *restrict Heap,    // Heap [1..nheap], not modified
    const int64_t *restrict Iheap,      // Iheap [0..n-1], not modified
    const int64_t n,                    // element names are in range 0:n-1
    const int64_t nheap                 // the number of nodes in the Heap
)
{

    char *msg = NULL ;
    LG_ASSERT_MSG (Heap != NULL && Iheap != NULL && nheap >= 0 && n >= 0, -2000,
        "Heap is invalid") ;

    // check all entries in the heap
    for (int64_t p = 1 ; p <= nheap ; p++)
    {
        LG_Element e = Heap [p] ;
        int64_t name = e.name ;
        LG_ASSERT_MSG (name >= 0 && name < n && p == Iheap [name], -2000,
            "Heap is invalid") ;
    }

    // check all objects
    for (int64_t name = 0 ; name < n ; name++)
    {
        int64_t p = Iheap [name] ;
        if (p <= 0)
        {
            // object with this name is not in the heap
        }
        else
        {
            LG_ASSERT_MSG (p <= nheap, -2000, "position of object is invalid") ;
            // object with this name is in the heap at position p
            LG_Element e = Heap [p] ;
            LG_ASSERT_MSG (e.name == name, -2000, "Heap is invalid") ;
        }
    }

    // Heap and Iheap are consistent
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// LG_heap_check: make sure the min-heap property holds for the whole Heap
//------------------------------------------------------------------------------

// Check the entire Heap to see if it has the min-heap property:  for all nodes
// in the Heap, the key of a node must less than or equal to the keys of its
// children  (duplicate keys may appear).  An empty Heap or a Heap of size 1
// always satisfies the min-heap property, but nheap < 0 is invalid.  This
// function is for assertions only.

static inline int LG_heap_check
(
    // input:
    const LG_Element *restrict Heap,    // Heap [1..nheap], not modified
    const int64_t *restrict Iheap,      // Iheap [0..n-1], not modified
    const int64_t n,                    // element names are in range 0:n-1
    const int64_t nheap                 // the number of nodes in the Heap
)
{

    char *msg = NULL ;
    LG_ASSERT_MSG (Heap != NULL && Iheap != NULL && nheap >= 0 && n >= 0, -2000,
        "Heap is invalid") ;

#if 0
    // dump the heap
    for (int64_t p = 1 ; p <= nheap ; p++)
    {
        printf ("Heap [%ld]: key %ld name: %ld\n", p, Heap [p].key,
            Heap [p].name) ;
        int64_t pleft  = 2*p ;          // left child of node p
        int64_t pright = pleft + 1 ;    // right child of node p
        if (pleft <= nheap)
        {
            printf ("     left  child: %ld (key %ld, name %ld)\n",
                pleft, Heap [pleft].key, Heap [pleft].name) ;
        }
        if (pright <= nheap)
        {
            printf ("     right child: %ld (key %ld, name %ld)\n",
                pright, Heap [pright].key, Heap [pright].name) ;
        }
        printf ("\n") ;
    }
#endif

    // nodes nheap/2 ... nheap have no children, so no need to check them
    for (int64_t p = 1 ; p <= nheap / 2 ; p++)
    {

        // consider node p.  Its key must be <= the key of both its children.

        int64_t pleft  = 2*p ;          // left child of node p
        int64_t pright = pleft + 1 ;    // right child of node p

        LG_ASSERT_MSG (! (pleft <= nheap && Heap [p].key > Heap [pleft].key),
            -2000, "the min-heap property is not satisfied") ;

        LG_ASSERT_MSG (! (pright <= nheap && Heap [p].key > Heap [pright].key),
            -2000, "the min-heap property is not satisfied") ;
    }

    // Heap satisfies the min-heap property; also check Iheap
    return (LG_iheap_check (Heap, Iheap, n, nheap)) ;
}

//------------------------------------------------------------------------------
// LG_heapify: enforce the min-heap property of a node
//------------------------------------------------------------------------------

// Heapify starting at node p in the Heap.  On input, the Heap rooted at node p
// satisfies the min-heap property, except for Heap [p] itself.  On output, all
// of the Heap rooted at node p satisfies the min-heap property.

static inline void LG_heapify
(
    int64_t p,                      // node that needs to be heapified
    LG_Element *restrict Heap,      // Heap [1..nheap]
    int64_t *restrict Iheap,        // Iheap [0..n-1]
    const int64_t n,                // max element name
    const int64_t nheap             // the number of nodes in the Heap
)
{

    //--------------------------------------------------------------------------
    // check inputs and check for quick return
    //--------------------------------------------------------------------------

    ASSERT (Heap != NULL && Iheap != NULL) ;

    if (p > nheap / 2 || nheap <= 1)
    {
        // nothing to do.  p has no children in the Heap.
        // Also safely do nothing if p is outside the Heap (p > nheap).
        return ;
    }

    //--------------------------------------------------------------------------
    // get the element to heapify
    //--------------------------------------------------------------------------

    // Get the element e at node p in the Heap; the one that needs heapifying.
    LG_Element e = Heap [p] ;

    // There is now a "hole" at Heap [p], with no element in it.

    //--------------------------------------------------------------------------
    // heapify
    //--------------------------------------------------------------------------

    while (true)
    {

        //----------------------------------------------------------------------
        // consider node p in the Heap
        //----------------------------------------------------------------------

        // Heap [p] is the "hole" in the Heap

        int64_t pleft  = 2*p ;          // left child of node p
        int64_t pright = pleft + 1 ;    // right child of node p

        if (pright <= nheap)
        {

            //------------------------------------------------------------------
            // both left and right children are in the Heap
            //------------------------------------------------------------------

            LG_Element eleft  = Heap [pleft] ;
            LG_Element eright = Heap [pright] ;
            if (eleft.key < eright.key)
            {
                // left node has a smaller key than the right node
                if (e.key > eleft.key)
                {
                    // key of element e is bigger than the left child of p, so
                    // bubble up the left child into the hole at Heap [p] and
                    // continue down the left child.  The hole moves to node
                    // pleft.
                    Heap [p] = eleft ;
                    Iheap [eleft.name] = p ;
                    p = pleft ;
                }
                else
                {
                    // done!  key of element e is is smaller than the left
                    // child of p; place e in the hole at p, and we're done.
                    Heap [p] = e ;
                    Iheap [e.name] = p ;
                    return ;
                }
            }
            else
            {
                // right node has a smaller key than the left node
                if (e.key > eright.key)
                {
                    // key of element e is bigger than the right child of p, so
                    // bubble up the right child into hole at Heap [p] and
                    // continue down the right child.  The hole moves to node
                    // pright.
                    Heap [p] = eright ;
                    Iheap [eright.name] = p ;
                    p = pright ;
                }
                else
                {
                    // done!  key of element e is is smaller than the right
                    // child of p; place e in the hole at p, and we're done.
                    Heap [p] = e ;
                    Iheap [e.name] = p ;
                    return ;
                }
            }
        }
        else
        {

            //------------------------------------------------------------------
            // right child is not in the Heap, see if left child is in the Heap
            //------------------------------------------------------------------

            if (pleft <= nheap)
            {
                // left child is in the Heap; check its key
                LG_Element eleft = Heap [pleft] ;
                if (e.key > eleft.key)
                {
                    // key of element e is bigger than the left child of p, so
                    // bubble up the left child into the hole at Heap [p] and
                    // continue down the left child.  The hole moves to node
                    // pleft.
                    Heap [p] = eleft ;
                    Iheap [eleft.name] = p ;
                    p = pleft ;
                }
            }

            //------------------------------------------------------------------
            // node p is a hole, and it has no children
            //------------------------------------------------------------------

            // put e in the hole, and we're done
            Heap [p] = e ;
            Iheap [e.name] = p ;
            return ;
        }
    }
}

//------------------------------------------------------------------------------
// LG_heap_build: construct a Heap
//------------------------------------------------------------------------------

// On input, the Heap [1..nheap] may not satisfy the min-heap property, but
// Iheap must already be initialized.
// If e = Heap [p], then Iheap [e.name] = p must hold.

// On output, the elements have been rearranged so that it satisfies the
// heap property.

static inline void LG_heap_build
(
    LG_Element *restrict Heap,      // Heap [1..nheap]
    int64_t *restrict Iheap,        // Iheap [0..n-1]
    const int64_t n,                // max element name
    const int64_t nheap             // the number of nodes in the Heap
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Heap != NULL && nheap >= 0) ;
    ASSERT (LG_iheap_check (Heap, Iheap, n, nheap)) ;

    //--------------------------------------------------------------------------
    // build the Heap
    //--------------------------------------------------------------------------

    for (int64_t p = nheap / 2 ; p >= 1 ; p--)
    {
        LG_heapify (p, Heap, Iheap, n, nheap) ;
    }

    //--------------------------------------------------------------------------
    // check result
    //--------------------------------------------------------------------------

    // Heap [1..nheap] now satisfies the min-heap property
    ASSERT (LG_heap_check (Heap, Iheap, n, nheap)) ;
}

//------------------------------------------------------------------------------
// LG_heap_delete: delete an element in the middle of a Heap
//------------------------------------------------------------------------------

static inline void LG_heap_delete
(
    int64_t p,                      // node that needs to be deleted
    LG_Element *restrict Heap,      // Heap [1..nheap]
    int64_t *restrict Iheap,        // Iheap [0..n-1]
    const int64_t n,                // max element name
    int64_t *restrict nheap         // the number of nodes in the Heap;
                                    // decremented on output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Heap != NULL && Iheap != NULL && (*nheap) >= 0) ;
    ASSERT (p >= 1 && p <= (*nheap)) ;

    //--------------------------------------------------------------------------
    // delete node p from the Heap
    //--------------------------------------------------------------------------

    // move the last node to node p and decrement the # of nodes in the Heap
    LG_Element elast = Heap [*nheap] ;  // last node in the heap
    LG_Element edel  = Heap [p] ;       // element to delete from the heap
    Heap [p] = elast ;              // move last node in the heap to position p
    Iheap [elast.name] = p ;        // elast has been moved to position p
    Iheap [edel.name] = 0 ;         // edel is no longer in the heap
    (*nheap)-- ;                    // one less entry in the heap

    // heapify node p (safely does nothing if node p was the one just deleted)
    LG_heapify (p, Heap, Iheap, n, (*nheap)) ;
}

//------------------------------------------------------------------------------
// LG_heap_decrease_key: decrease the key of an entry in the heap
//------------------------------------------------------------------------------

static inline void LG_heap_decrease_key
(
    int64_t p,                      // entry to modify in the heap
    const LG_key_t new_key,         // new key value of Heap [p]
    LG_Element *restrict Heap,      // Heap [1..nheap]
    int64_t *restrict Iheap,        // Iheap [0..n-1]
    const int64_t n,                // max element name
    const int64_t nheap             // the number of nodes in the Heap
)
{
    ASSERT (Heap != NULL && Iheap != NULL) ;
    ASSERT (p >= 1 && p < nheap) ;
    ASSERT (new_key < Heap [p].key) ;

//  printf ("Decreasing Heap [%ld] = name: %ld key: from %ld to %ld\n",
//      p, Heap [p].name, Heap [p].key, new_key) ;

    Heap [p].key = new_key ;
    int64_t parent = p/2 ;

    while (p > 1 && Heap [parent].key > Heap [p].key)
    {
        // swap Heap [p] and Heap [parent]
//      printf ("swap positions %ld and %ld\n", p, parent) ;

        LG_Element e = Heap [p] ;
        Heap [p] = Heap [parent] ;
        Heap [parent] = e ;

        // update the inverse heap
        Iheap [Heap [p].name] = p ;
        Iheap [Heap [parent].name] = parent ;

        // advance up to the parent
        p = parent ;
        parent = p/2 ;
    }
}

#endif
