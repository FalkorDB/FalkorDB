//------------------------------------------------------------------------------
// GB_binary_search.h: binary search in a sorted list
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These methods are used for all kernels, include CUDA kernels, which is why
// GB_STATIC_INLINE is used.  It becomes "static inline" on the CPU, and
// "static __device__ inline" on the GPU.

#ifndef GB_BINARY_SEARCH_H
#define GB_BINARY_SEARCH_H

//------------------------------------------------------------------------------
// GB_trim_binary_search: simple binary search
//------------------------------------------------------------------------------

// search for integer i in the list X [pleft...pright]; no zombies.
// The list X [pleft ... pright] is in ascending order.  It may have
// duplicates.

GB_STATIC_INLINE void GB_trim_binary_search_32
(
    const uint32_t i,           // item to look for
    const uint32_t *restrict X, // array to search; no zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    // binary search of X [pleft ... pright] for the integer i
    while (*pleft < *pright)
    {
        // Both of following methods work on both the GPU and CPU, but the
        // first method works fastest on the GPU, while the 2nd works fastest
        // on the CPU.
        #ifdef GB_CUDA_KERNEL
            // binary search on the GPU
            int64_t pmiddle = (*pleft + *pright) >> 1 ;
            bool less = (X [pmiddle] < i) ;
            *pleft  = less ? (pmiddle+1) : *pleft ;
            *pright = less ? *pright : pmiddle ;
        #else
            // binary search on the CPU
            int64_t pmiddle = (*pleft + *pright) / 2 ;
            if (X [pmiddle] < i)
            {
                // if in the list, it appears in [pmiddle+1..pright]
                *pleft = pmiddle + 1 ;
            }
            else
            {
                // if in the list, it appears in [pleft..pmiddle]
                *pright = pmiddle ;
            }
        #endif
    }
    // binary search is narrowed down to a single item
    // or it has found the list is empty
    ASSERT (*pleft == *pright || *pleft == *pright + 1) ;
}

GB_STATIC_INLINE void GB_trim_binary_search_64
(
    const uint64_t i,           // item to look for
    const uint64_t *restrict X, // array to search; no zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    // binary search of X [pleft ... pright] for the integer i
    while (*pleft < *pright)
    {
        #ifdef GB_CUDA_KERNEL
            int64_t pmiddle = (*pleft + *pright) >> 1 ;
            bool less = (X [pmiddle] < i) ;
            *pleft  = less ? (pmiddle+1) : *pleft ;
            *pright = less ? *pright : pmiddle ;
        #else
            int64_t pmiddle = (*pleft + *pright) / 2 ;
            if (X [pmiddle] < i)
            {
                // if in the list, it appears in [pmiddle+1..pright]
                *pleft = pmiddle + 1 ;
            }
            else
            {
                // if in the list, it appears in [pleft..pmiddle]
                *pright = pmiddle ;
            }
        #endif
    }
    // binary search is narrowed down to a single item
    // or it has found the list is empty
    ASSERT (*pleft == *pright || *pleft == *pright + 1) ;
}

GB_STATIC_INLINE void GB_trim_binary_search
(
    const uint64_t i,           // item to look for
    const void *X,              // array to search; no zombies
    const bool X_is_32,         // if true, X is 32-bit, else 64
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    if (X_is_32)
    {
        GB_trim_binary_search_32 (i, (const uint32_t *) X, pleft, pright) ;
    }
    else
    {
        GB_trim_binary_search_64 (i, (const uint64_t *) X, pleft, pright) ;
    }
}

//------------------------------------------------------------------------------
// GB_binary_search: binary search and check if found
//------------------------------------------------------------------------------

// If found is true then X [pleft == pright] == i.  If duplicates appear then
// X [pleft] is any one of the entries with value i in the list.
// If found is false then
//    X [original_pleft ... pleft-1] < i and
//    X [pleft+1 ... original_pright] > i holds.
// The value X [pleft] may be either < or > i.

GB_STATIC_INLINE bool GB_binary_search_32
(
    const uint32_t i,           // item to look for
    const uint32_t *restrict X, // array to search; no zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    GB_trim_binary_search_32 (i, X, pleft, pright) ;
    return (*pleft == *pright && X [*pleft] == i) ;
}

GB_STATIC_INLINE bool GB_binary_search_64
(
    const uint64_t i,           // item to look for
    const uint64_t *restrict X, // array to search; no zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    GB_trim_binary_search_64 (i, X, pleft, pright) ;
    return (*pleft == *pright && X [*pleft] == i) ;
}

GB_STATIC_INLINE bool GB_binary_search
(
    const uint64_t i,           // item to look for
    const void *X,              // array to search; no zombies
    const bool X_is_32,         // if true, X is 32-bit, else 64
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    if (X_is_32)
    {
        return (GB_binary_search_32 (i, (const uint32_t *) X, pleft, pright)) ;
    }
    else
    {
        return (GB_binary_search_64 (i, (const uint64_t *) X, pleft, pright)) ;
    }
}

//------------------------------------------------------------------------------
// GB_split_binary_search: binary search, and then partition the list
//------------------------------------------------------------------------------

// If found is true then X [pleft] == i.  If duplicates appear then X [pleft]
//    is any one of the entries with value i in the list.
// If found is false then
//    X [original_pleft ... pleft-1] < i and
//    X [pleft ... original_pright] > i holds, and pleft-1 == pright
// If X has no duplicates, then whether or not i is found,
//    X [original_pleft ... pleft-1] < i and
//    X [pleft ... original_pright] >= i holds.

GB_STATIC_INLINE bool GB_split_binary_search_32
(
    const uint32_t i,           // item to look for
    const uint32_t *restrict X, // array to search; no zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    bool found = GB_binary_search_32 (i, X, pleft, pright) ;
    if (!found && (*pleft == *pright))
    {
        if (i > X [*pleft])
        {
            (*pleft)++ ;
        }
        else
        {
            (*pright)++ ;
        }
    }
    return (found) ;
}

GB_STATIC_INLINE bool GB_split_binary_search_64
(
    const uint64_t i,           // item to look for
    const uint64_t *restrict X, // array to search; no zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    bool found = GB_binary_search_64 (i, X, pleft, pright) ;
    if (!found && (*pleft == *pright))
    {
        if (i > X [*pleft])
        {
            (*pleft)++ ;
        }
        else
        {
            (*pright)++ ;
        }
    }
    return (found) ;
}

GB_STATIC_INLINE bool GB_split_binary_search
(
    const uint64_t i,           // item to look for
    const void *X,              // array to search; no zombies
    const bool X_is_32,         // if true, X is 32-bit, else 64
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    if (X_is_32)
    {
        return (GB_split_binary_search_32 (i, (const uint32_t *) X,
            pleft, pright)) ;
    }
    else
    {
        return (GB_split_binary_search_64 (i, (const uint64_t *) X,
            pleft, pright)) ;
    }
}

//------------------------------------------------------------------------------
// GB_trim_binary_search_zombie: binary search in the presence of zombies
//------------------------------------------------------------------------------

GB_STATIC_INLINE void GB_trim_binary_search_zombie_32
(
    const uint32_t i,           // item to look for
    const int32_t *restrict X,  // array to search; with zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    // binary search of X [pleft ... pright] for the integer i
    while (*pleft < *pright)
    {
        #ifdef GB_CUDA_KERNEL
            int64_t pmiddle = (*pleft + *pright) >> 1 ;
            int64_t ix = X [pmiddle] ;
            ix = GB_UNZOMBIE (ix) ;
            bool less = (ix < i) ;
            *pleft  = less ? (pmiddle+1) : *pleft ;
            *pright = less ? *pright : pmiddle ;
        #else
            int64_t pmiddle = (*pleft + *pright) / 2 ;
            int64_t ix = X [pmiddle] ;
            ix = GB_UNZOMBIE (ix) ;
            if (ix < i)
            {
                // if in the list, it appears in [pmiddle+1..pright]
                *pleft = pmiddle + 1 ;
            }
            else
            {
                // if in the list, it appears in [pleft..pmiddle]
                *pright = pmiddle ;
            }
        #endif
    }
    // binary search is narrowed down to a single item
    // or it has found the list is empty
    ASSERT (*pleft == *pright || *pleft == *pright + 1) ;
}

GB_STATIC_INLINE void GB_trim_binary_search_zombie_64
(
    const uint64_t i,           // item to look for
    const int64_t *restrict X,  // array to search; with zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright
)
{
    // binary search of X [pleft ... pright] for the integer i
    while (*pleft < *pright)
    {
        #ifdef GB_CUDA_KERNEL
            int64_t pmiddle = (*pleft + *pright) >> 1 ;
            int64_t ix = X [pmiddle] ;
            ix = GB_UNZOMBIE (ix) ;
            bool less = (ix < i) ;
            *pleft  = less ? (pmiddle+1) : *pleft ;
            *pright = less ? *pright : pmiddle ;
        #else
            int64_t pmiddle = (*pleft + *pright) / 2 ;
            int64_t ix = X [pmiddle] ;
            ix = GB_UNZOMBIE (ix) ;
            if (ix < i)
            {
                // if in the list, it appears in [pmiddle+1..pright]
                *pleft = pmiddle + 1 ;
            }
            else
            {
                // if in the list, it appears in [pleft..pmiddle]
                *pright = pmiddle ;
            }
        #endif
    }
    // binary search is narrowed down to a single item
    // or it has found the list is empty
    ASSERT (*pleft == *pright || *pleft == *pright + 1) ;
}

//------------------------------------------------------------------------------
// GB_binary_search_zombie: binary search with zombies; check if found
//------------------------------------------------------------------------------

GB_STATIC_INLINE bool GB_binary_search_zombie_32
(
    const uint32_t i,           // item to look for
    const int32_t *restrict X,  // array to search; with zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright,
    const bool may_see_zombies,
    bool *is_zombie
)
{
    bool found = false ;
    *is_zombie = false ;
    if (may_see_zombies)
    {
        GB_trim_binary_search_zombie_32 (i, X, pleft, pright) ;
        if (*pleft == *pright)
        {
            int64_t i2 = X [*pleft] ;
            *is_zombie = GB_IS_ZOMBIE (i2) ;
            if (*is_zombie) i2 = GB_DEZOMBIE (i2) ;
            found = (i == i2) ;
        }
    }
    else
    {
        found = GB_binary_search_32 (i, (const uint32_t *) X, pleft, pright) ;
    }
    return (found) ;
}

GB_STATIC_INLINE bool GB_binary_search_zombie_64
(
    const uint64_t i,           // item to look for
    const int64_t *restrict X,  // array to search; with zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright,
    const bool may_see_zombies,
    bool *is_zombie
)
{
    bool found = false ;
    *is_zombie = false ;
    if (may_see_zombies)
    {
        GB_trim_binary_search_zombie_64 (i, X, pleft, pright) ;
        if (*pleft == *pright)
        {
            int64_t i2 = X [*pleft] ;
            *is_zombie = GB_IS_ZOMBIE (i2) ;
            if (*is_zombie) i2 = GB_DEZOMBIE (i2) ;
            found = (i == i2) ;
        }
    }
    else
    {
        found = GB_binary_search_64 (i, (const uint64_t *) X, pleft, pright) ;
    }
    return (found) ;
}

GB_STATIC_INLINE bool GB_binary_search_zombie
(
    const uint64_t i,           // item to look for
    const void *X,              // array to search; with zombies
    const bool X_is_32,         // if true, X is 32-bit, else 64
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright,
    const bool may_see_zombies,
    bool *is_zombie
)
{
    if (X_is_32)
    {
        return (GB_binary_search_zombie_32 (i, (const int32_t *) X,
            pleft, pright, may_see_zombies, is_zombie)) ;
    }
    else
    {
        return (GB_binary_search_zombie_64 (i, (const int64_t *) X,
            pleft, pright, may_see_zombies, is_zombie)) ;
    }
}

//------------------------------------------------------------------------------
// GB_split_binary_search_zombie: binary search with zombies; then partition
//------------------------------------------------------------------------------

GB_STATIC_INLINE bool GB_split_binary_search_zombie_32
(
    const uint32_t i,           // item to look for
    const int32_t *restrict X,  // array to search; with zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright,
    const bool may_see_zombies,
    bool *is_zombie
)
{
    bool found = false ;
    *is_zombie = false ;
    if (may_see_zombies)
    {
        GB_trim_binary_search_zombie_32 (i, X, pleft, pright) ;
        if (*pleft == *pright)
        {
            int64_t i2 = X [*pleft] ;
            *is_zombie = GB_IS_ZOMBIE (i2) ;
            if (*is_zombie) i2 = GB_DEZOMBIE (i2) ;
            found = (i == i2) ;
            if (!found)
            {
                if (i > i2)
                {
                    (*pleft)++ ;
                }
                else
                {
                    (*pright)++ ;
                }
            }
        }
    }
    else
    {
        found = GB_split_binary_search_32 (i, (const uint32_t *) X,
            pleft, pright) ;
    }
    return (found) ;
}

GB_STATIC_INLINE bool GB_split_binary_search_zombie_64
(
    const uint64_t i,           // item to look for
    const int64_t *restrict X,  // array to search; with zombies
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright,
    const bool may_see_zombies,
    bool *is_zombie
)
{
    bool found = false ;
    *is_zombie = false ;
    if (may_see_zombies)
    {
        GB_trim_binary_search_zombie_64 (i, X, pleft, pright) ;
        if (*pleft == *pright)
        {
            int64_t i2 = X [*pleft] ;
            *is_zombie = GB_IS_ZOMBIE (i2) ;
            if (*is_zombie) i2 = GB_DEZOMBIE (i2) ;
            found = (i == i2) ;
            if (!found)
            {
                if (i > i2)
                {
                    (*pleft)++ ;
                }
                else
                {
                    (*pright)++ ;
                }
            }
        }
    }
    else
    {
        found = GB_split_binary_search_64 (i, (const uint64_t *) X,
            pleft, pright) ;
    }
    return (found) ;
}

GB_STATIC_INLINE bool GB_split_binary_search_zombie
(
    const uint64_t i,           // item to look for
    const void *X,              // array to search; with zombies
    const bool X_is_32,         // if true, X is 32-bit, else 64
    int64_t *pleft,             // look in X [pleft:pright]
    int64_t *pright,
    const bool may_see_zombies,
    bool *is_zombie
)
{
    if (X_is_32)
    {
        return (GB_split_binary_search_zombie_32 (i, (const int32_t *) X,
            pleft, pright, may_see_zombies, is_zombie)) ;
    }
    else
    {
        return (GB_split_binary_search_zombie_64 (i, (const int64_t *) X,
            pleft, pright, may_see_zombies, is_zombie)) ;
    }
}

#endif

