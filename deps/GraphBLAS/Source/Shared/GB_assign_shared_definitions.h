//------------------------------------------------------------------------------
// GB_assign_shared_definitions.h: definitions for GB_subassign kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// macros for the construction of the GB_subassign kernels

#include "GB_kernel_shared_definitions.h"

//------------------------------------------------------------------------------
// definitions redefined as needed
//------------------------------------------------------------------------------

#ifndef GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE ;
#endif

#undef  GB_FREE_ALL
#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    GB_WERK_POP (Npending, int64_t) ;           \
    GB_FREE_WORK (&TaskList, TaskList_size) ;   \
    GB_FREE (&Zh, Zh_size) ;                    \
    GB_FREE_WORK (&Z_to_X, Z_to_X_size) ;       \
    GB_FREE_WORK (&Z_to_S, Z_to_S_size) ;       \
    GB_FREE_WORK (&Z_to_A, Z_to_A_size) ;       \
    GB_FREE_WORK (&Z_to_M, Z_to_M_size) ;       \
    GB_Matrix_free (&S) ;                       \
}

//------------------------------------------------------------------------------
// definitions done just once
//------------------------------------------------------------------------------

#ifndef GB_SUBASSIGN_SHARED_DEFINITIONS_H
#define GB_SUBASSIGN_SHARED_DEFINITIONS_H

//------------------------------------------------------------------------------
// GB_EMPTY_TASKLIST: declare an empty TaskList
//------------------------------------------------------------------------------

#define GB_EMPTY_TASKLIST                                                   \
    GrB_Info info ;                                                         \
    int taskid, ntasks = 0, nthreads ;                                      \
    GB_task_struct *TaskList = NULL ; size_t TaskList_size = 0 ;            \
    GB_WERK_DECLARE (Npending, int64_t) ;                                   \
    int64_t *restrict Zh     = NULL ; size_t Zh_size = 0 ;                  \
    int64_t *restrict Z_to_X = NULL ; size_t Z_to_X_size = 0 ;              \
    int64_t *restrict Z_to_S = NULL ; size_t Z_to_S_size = 0 ;              \
    int64_t *restrict Z_to_A = NULL ; size_t Z_to_A_size = 0 ;              \
    int64_t *restrict Z_to_M = NULL ; size_t Z_to_M_size = 0 ;              \
    struct GB_Matrix_opaque S_header ;                                      \
    GrB_Matrix S = NULL ;

//------------------------------------------------------------------------------
// GB_GET_C: get the C matrix (cannot be bitmap)
//------------------------------------------------------------------------------

// C cannot be aliased with M or A.

#define GB_GET_C                                                            \
    ASSERT_MATRIX_OK (C, "C for subassign kernel", GB0) ;                   \
    ASSERT (!GB_IS_BITMAP (C)) ;                                            \
    const bool C_iso = C->iso ;                                             \
    int64_t *restrict Ci = C->i ;                                           \
    GB_void *restrict Cx = (C_iso) ? NULL : (GB_void *) C->x ;              \
    const size_t csize = C->type->size ;                                    \
    const GB_Type_code ccode = C->type->code ;                              \
    const int64_t cvdim = C->vdim ;                                         \
    const int64_t Cvlen = C->vlen ;                                         \
    int64_t nzombies = C->nzombies ;                                        \
    const bool is_matrix = (cvdim > 1) ;

#ifndef GB_DECLAREC
#define GB_DECLAREC(cwork) GB_void cwork [GB_VLA(csize)] ;
#endif

#define GB_GET_C_HYPER_HASH                                                 \
    GB_OK (GB_hyper_hash_build (C, Werk)) ;                                 \
    const int64_t *restrict C_Yp = (C->Y == NULL) ? NULL : C->Y->p ;        \
    const int64_t *restrict C_Yi = (C->Y == NULL) ? NULL : C->Y->i ;        \
    const int64_t *restrict C_Yx = (C->Y == NULL) ? NULL : C->Y->x ;        \
    const int64_t C_hash_bits = (C->Y == NULL) ? 0 : (C->Y->vdim - 1) ;

//------------------------------------------------------------------------------
// GB_GET_MASK: get the mask matrix M
//------------------------------------------------------------------------------

// M and A can be aliased, but both are const.

#define GB_GET_M                                                            \
    ASSERT_MATRIX_OK (M, "mask M", GB0) ;                                   \
    const int64_t *Mp = M->p ;                                              \
    const int8_t  *Mb = M->b ;                                              \
    const int64_t *Mh = M->h ;                                              \
    const int64_t *Mi = M->i ;                                              \
    const GB_M_TYPE *Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;     \
    const size_t msize = M->type->size ;                                    \
    const size_t Mvlen = M->vlen ;

#define GB_GET_MASK                                                         \
    GB_GET_M ;                                                              \
    const int64_t Mnvec = M->nvec ;                                         \
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;                         \
    const bool M_is_bitmap = GB_IS_BITMAP (M)

#define GB_GET_MASK_HYPER_HASH                                              \
    GB_OK (GB_hyper_hash_build (M, Werk)) ;                                 \
    const int64_t *restrict M_Yp = (M->Y == NULL) ? NULL : M->Y->p ;        \
    const int64_t *restrict M_Yi = (M->Y == NULL) ? NULL : M->Y->i ;        \
    const int64_t *restrict M_Yx = (M->Y == NULL) ? NULL : M->Y->x ;        \
    const int64_t M_hash_bits = (M->Y == NULL) ? 0 : (M->Y->vdim - 1) ;

//------------------------------------------------------------------------------
// GB_GET_ACCUM: get the accumulator op and its related typecasting functions
//------------------------------------------------------------------------------

#define GB_GET_ACCUM                                                        \
    ASSERT_BINARYOP_OK (accum, "accum for assign", GB0) ;                   \
    ASSERT (!GB_OP_IS_POSITIONAL (accum)) ;                                 \
    const GxB_binary_function faccum = accum->binop_function ;              \
    const GB_cast_function                                                  \
        cast_A_to_Y = GB_cast_factory (accum->ytype->code, acode),          \
        cast_C_to_X = GB_cast_factory (accum->xtype->code, ccode),          \
        cast_Z_to_C = GB_cast_factory (ccode, accum->ztype->code) ;         \
    const size_t xsize = accum->xtype->size ;                               \
    const size_t ysize = accum->ytype->size ;                               \
    const size_t zsize = accum->ztype->size ;

#ifndef GB_DECLAREZ
#define GB_DECLAREZ(zwork) GB_void zwork [GB_VLA(zsize)] ;
#endif

#ifndef GB_DECLAREX
#define GB_DECLAREX(xwork) GB_void xwork [GB_VLA(xsize)] ;
#endif

#ifndef GB_DECLAREY
#define GB_DECLAREY(ywork) GB_void ywork [GB_VLA(ysize)] ;
#endif

//------------------------------------------------------------------------------
// GB_GET_A: get the A matrix
//------------------------------------------------------------------------------

#define GB_GET_A                                                            \
    ASSERT_MATRIX_OK (A, "A for assign", GB0) ;                             \
    const GrB_Type atype = A->type ;                                        \
    const size_t asize = atype->size ;                                      \
    const int64_t *Ap = A->p ;                                              \
    const int8_t  *Ab = A->b ;                                              \
    const int64_t *Ai = A->i ;                                              \
    const int64_t Avlen = A->vlen ;                                         \
    const GB_void *Ax = (GB_void *) A->x ;                                  \
    const bool A_iso = A->iso ;                                             \
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;                             \
    const GB_Type_code acode = atype->code ;                                \
    const GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;   \
    GB_DECLAREC (cwork) ;                                                   \
    if (A_iso)                                                              \
    {                                                                       \
        /* cwork = (ctype) Ax [0], typecast iso value of A into cwork */    \
        cast_A_to_C (cwork, Ax, asize) ;                                    \
    }

#ifndef GB_DECLAREA
#define GB_DECLAREA(awork) GB_void awork [GB_VLA(asize)] ;
#endif

//------------------------------------------------------------------------------
// GB_GET_SCALAR: get the scalar
//------------------------------------------------------------------------------

#define GB_GET_SCALAR                                                       \
    const GrB_Type atype = scalar_type ;                                    \
    ASSERT_TYPE_OK (atype, "atype for assign", GB0) ;                       \
    const size_t asize = atype->size ;                                      \
    const GB_Type_code acode = atype->code ;                                \
    const GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;   \
    GB_DECLAREC (cwork) ;                                                   \
    cast_A_to_C (cwork, scalar, asize) ;

//------------------------------------------------------------------------------
// GB_GET_ACCUM_SCALAR: get the scalar and the accumulator
//------------------------------------------------------------------------------

#define GB_GET_ACCUM_SCALAR                                                 \
    GB_GET_SCALAR ;                                                         \
    GB_GET_ACCUM ;                                                          \
    GB_DECLAREY (ywork) ;                                                   \
    cast_A_to_Y (ywork, scalar, asize) ;

#define GB_GET_ACCUM_MATRIX                                                 \
    GB_GET_A ;                                                              \
    GB_GET_ACCUM ;                                                          \
    GB_DECLAREY (ywork) ;                                                   \
    if (A_iso)                                                              \
    {                                                                       \
        cast_A_to_Y (ywork, Ax, asize) ;                                    \
    }

//------------------------------------------------------------------------------
// GB_GET_S: get the S matrix
//------------------------------------------------------------------------------

// S is never aliased with any other matrix.
// FUTURE: S->p could be C->p and S->x NULL if I and J are (:,:)

#define GB_GET_S                                                            \
    ASSERT_MATRIX_OK (S, "S extraction", GB0) ;                             \
    const int64_t *restrict Sp = S->p ;                                     \
    const int64_t *restrict Sh = S->h ;                                     \
    const int64_t *restrict Si = S->i ;                                     \
    const int64_t *restrict Sx = (int64_t *) S->x ;                         \
    const int64_t Svlen = S->vlen ;                                         \
    const int64_t Snvec = S->nvec ;                                         \
    const bool S_is_hyper = GB_IS_HYPERSPARSE (S) ;                         \
    const int64_t *restrict S_Yp = (S->Y == NULL) ? NULL : S->Y->p ;        \
    const int64_t *restrict S_Yi = (S->Y == NULL) ? NULL : S->Y->i ;        \
    const int64_t *restrict S_Yx = (S->Y == NULL) ? NULL : S->Y->x ;        \
    const int64_t S_hash_bits = (S->Y == NULL) ? 0 : (S->Y->vdim - 1) ;

//------------------------------------------------------------------------------
// basic actions
//------------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // S_Extraction: finding C(iC,jC) via lookup through S=C(I,J)
    //--------------------------------------------------------------------------

    // S is the symbolic pattern of the submatrix S = C(I,J).  The "numerical"
    // value (held in S->x) of an entry S(i,j) is not a value, but a pointer
    // back into C where the corresponding entry C(iC,jC) can be found, where
    // iC = I [i] and jC = J [j].

    // The following macro performs the lookup.  Given a pointer pS into a
    // column S(:,j), it finds the entry C(iC,jC), and also determines if the
    // C(iC,jC) entry is a zombie.  The column indices j and jC are implicit.

    // Used for Methods 00 to 04, 06s, and 09 to 20, all of which use S.

    #define GB_C_S_LOOKUP                                                   \
        int64_t pC = Sx [pS] ;                                              \
        int64_t iC = GBI (Ci, pC, Cvlen) ;                                  \
        bool is_zombie = GB_IS_ZOMBIE (iC) ;                                \
        if (is_zombie) iC = GB_FLIP (iC) ;

    //--------------------------------------------------------------------------
    // C(:,jC) is dense: iC = I [iA], and then look up C(iC,jC)
    //--------------------------------------------------------------------------

    // C(:,jC) is dense, and thus can be accessed with a O(1)-time lookup
    // with the index iC, where the index iC comes from I [iA] or via a
    // colon notation for I.

    // This used for Methods 05, 06n, 07, and 08n, which do not use S.

    #define GB_iC_DENSE_LOOKUP                                              \
        int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;                     \
        int64_t pC = pC_start + iC ;                                        \
        bool is_zombie = (Ci != NULL) && GB_IS_ZOMBIE (Ci [pC]) ;           \
        ASSERT (GB_IMPLIES (Ci != NULL, GB_UNFLIP (Ci [pC]) == iC)) ;

    //--------------------------------------------------------------------------
    // get C(iC,jC) via binary search of C(:,jC)
    //--------------------------------------------------------------------------

    // This used for Methods 05, 06n, 07, and 08n, which do not use S.

    // New zombies may be introduced into C during the parallel computation.
    // No coarse task shares the same C(:,jC) vector, so no race condition can
    // occur.  Fine tasks do share the same C(:,jC) vector, but each fine task
    // is given a unique range of pC_start:pC_end-1 to search.  Thus, no binary
    // search of any fine tasks conflict with each other.

    #define GB_iC_BINARY_SEARCH                                             \
        int64_t iC = GB_ijlist (I, iA, Ikind, Icolon) ;                     \
        int64_t pC = pC_start ;                                             \
        int64_t pright = pC_end - 1 ;                                       \
        bool cij_found, is_zombie ;                                         \
        GB_BINARY_SEARCH_ZOMBIE (iC, Ci, pC, pright, cij_found, zorig,      \
            is_zombie) ;

    //--------------------------------------------------------------------------
    // for a 2-way or 3-way merge
    //--------------------------------------------------------------------------

    // An entry S(i,j), A(i,j), or M(i,j) has been processed;
    // move to the next one.
    #define GB_NEXT(X) (p ## X)++ ;

    //--------------------------------------------------------------------------
    // basic operations
    //--------------------------------------------------------------------------

    #ifndef GB_COPY_scalar_to_C
    #define GB_COPY_scalar_to_C(Cx,pC,cwork)                                \
    {                                                                       \
        /* C(iC,jC) = scalar, already typecasted into cwork */              \
        if (!C_iso)                                                         \
        {                                                                   \
            memcpy (Cx +((pC)*csize), cwork, csize) ;                       \
        }                                                                   \
    }
    #endif

    #ifndef GB_COPY_aij_to_C
    #define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork)                       \
    {                                                                       \
        /* C(iC,jC) = (ctype) A(i,j), with typecasting */                   \
        if (!C_iso)                                                         \
        {                                                                   \
            if (A_iso)                                                      \
            {                                                               \
                /* cwork = (ctype) Ax [0], A iso value already done */      \
                memcpy (Cx +((pC)*csize), cwork, csize) ;                   \
            }                                                               \
            else                                                            \
            {                                                               \
                cast_A_to_C (Cx +(pC*csize), Ax +(pA*asize), asize) ;       \
            }                                                               \
        }                                                                   \
    }
    #endif

    #ifndef GB_COPY_aij_to_cwork
    #define GB_COPY_aij_to_cwork(cwork,Ax,pA,A_iso)                         \
    {                                                                       \
        /* cwork = (ctype) A(i,j), with typecasting */                      \
        if (!C_iso)                                                         \
        {                                                                   \
            cast_A_to_C (cwork, Ax +(A_iso ? 0: (pA)*asize), asize) ;       \
        }                                                                   \
    }

    #endif

    #ifndef GB_COPY_aij_to_ywork
    #define GB_COPY_aij_to_ywork(ywork,Ax,pA,A_iso)                         \
    {                                                                       \
        /* ywork = A(i,j), with typecasting */                              \
        if (!C_iso)                                                         \
        {                                                                   \
            cast_A_to_Y (ywork, Ax + (A_iso ? 0 : (pA*asize)), asize) ;     \
        }                                                                   \
    }
    #endif

    #ifndef GB_ACCUMULATE_scalar
    #define GB_ACCUMULATE_scalar(Cx,pC,ywork)                               \
    {                                                                       \
        if (!C_iso)                                                         \
        {                                                                   \
            /* C(iC,jC) += ywork, with typecasting */                       \
            GB_DECLAREX (xwork) ;                                           \
            cast_C_to_X (xwork, Cx +(pC*csize), csize) ;                    \
            GB_DECLAREZ (zwork) ;                                           \
            faccum (zwork, xwork, ywork) ;                                  \
            cast_Z_to_C (Cx +(pC*csize), zwork, csize) ;                    \
        }                                                                   \
    }
    #endif

    #ifndef GB_ACCUMULATE_aij
    #define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork)                      \
    {                                                                       \
        /* Cx [pC] += (ytype) Ax [A_iso ? 0 : pA] */                        \
        if (!C_iso)                                                         \
        {                                                                   \
            /* xwork = (xtype) Cx [pC] */                                   \
            GB_DECLAREX (xwork) ;                                           \
            cast_C_to_X (xwork, Cx +(pC*csize), csize) ;                    \
            GB_DECLAREZ (zwork) ;                                           \
            if (A_iso)                                                      \
            {                                                               \
                /* zwork = op (xwork, ywork) */                             \
                faccum (zwork, xwork, ywork) ;                              \
            }                                                               \
            else                                                            \
            {                                                               \
                /* ywork = (ytype) A(i,j) */                                \
                GB_DECLAREY (ywork) ;                                       \
                cast_A_to_Y (ywork, Ax + (pA*asize), asize) ;               \
                /* zwork = op (xwork, ywork) */                             \
                faccum (zwork, xwork, ywork) ;                              \
            }                                                               \
            /* Cx [pC] = (ctype) zwork */                                   \
            cast_Z_to_C (Cx +(pC*csize), zwork, csize) ;                    \
        }                                                                   \
    }
    #endif

    #define GB_DELETE                                                       \
    {                                                                       \
        /* turn C(iC,jC) into a zombie */                                   \
        ASSERT (!GB_IS_FULL (C)) ;                                          \
        task_nzombies++ ;                                                   \
        Ci [pC] = GB_FLIP (iC) ;                                            \
    }

    #define GB_UNDELETE                                                     \
    {                                                                       \
        /* bring a zombie C(iC,jC) back to life;                 */         \
        /* the value of C(iC,jC) must also be assigned.          */         \
        ASSERT (!GB_IS_FULL (C)) ;                                          \
        Ci [pC] = iC ;                                                      \
        task_nzombies-- ;                                                   \
    }

    //--------------------------------------------------------------------------
    // C(I,J)<M> = accum (C(I,J),A): consider all cases
    //--------------------------------------------------------------------------

        // The matrix C may have pending tuples and zombies:

        // (1) pending tuples:  this is a list of pending updates held as a set
        // of (i,j,x) tuples.  They had been added to the list via a prior
        // GrB_setElement or GxB_subassign.  No operator needs to be applied to
        // them; the implied operator is SECOND, for both GrB_setElement and
        // GxB_subassign, regardless of whether or not an accum operator is
        // present.  Pending tuples are inserted if and only if the
        // corresponding entry C(i,j) does not exist, and in that case no accum
        // operator is applied.

        //      The GrB_setElement method (C(i,j) = x) is same as GxB_subassign
        //      with: accum is SECOND, C not replaced, no mask M, mask not
        //      complemented.  If GrB_setElement needs to insert its update as
        //      a pending tuple, then it will always be compatible with all
        //      pending tuples inserted here, by GxB_subassign.

        // (2) zombie entries.  These are entries that are still present in the
        // pattern but marked for deletion (via GB_FLIP(i) for the row index).

        // For the current GxB_subassign, there are 16 cases to handle,
        // all combinations of the following options:

        //      accum is NULL, accum is not NULL
        //      C is not replaced, C is replaced
        //      no mask, mask is present
        //      mask is not complemented, mask is complemented

        // Complementing an empty mask:  This does not require the matrix A
        // at all so it is handled as a special case.  It corresponds to
        // the GB_RETURN_IF_QUICK_MASK option in other GraphBLAS operations.
        // Thus only 12 cases are considered in the tables below:

        //      These 4 cases are listed in Four Tables below:
        //      2 cases: accum is NULL, accum is not NULL
        //      2 cases: C is not replaced, C is replaced

        //      3 cases: no mask, M is present and not complemented,
        //               and M is present and complemented.  If there is no
        //               mask, then M(i,j)=1 for all (i,j).  These 3 cases
        //               are the columns of each of the Four Tables.

        // Each of these 12 cases can encounter up to 12 combinations of
        // entries in C, A, and M (6 if no mask M is present).  The left
        // column of the Four Tables below consider all 12 combinations for all
        // (i,j) in the cross product IxJ:

        //      C(I(i),J(j)) present, zombie, or not there: C, X, or '.'
        //      A(i,j) present or not, labeled 'A' or '.' below
        //      M(i,j) = 1 or 0 (but only if M is present)

        //      These 12 cases become the left columns as listed below.
        //      The zombie cases are handled a sub-case for "C present:
        //      regular entry or zombie".  The acronyms below use "D" for
        //      "dot", meaning the entry (C or A) is not present.

        //      [ C A 1 ]   C_A_1: both C and A present, M=1
        //      [ X A 1 ]   C_A_1: both C and A present, M=1, C is a zombie
        //      [ . A 1 ]   D_A_1: C not present, A present, M=1

        //      [ C . 1 ]   C_D_1: C present, A not present, M=1
        //      [ X . 1 ]   C_D_1: C present, A not present, M=1, C a zombie
        //      [ . . 1 ]          only M=1 present, but nothing to do

        //      [ C A 0 ]   C_A_0: both C and A present, M=0
        //      [ X A 0 ]   C_A_0: both C and A present, M=0, C is a zombie
        //      [ . A 0 ]          C not present, A present, M=0,
        //                              nothing to do

        //      [ C . 0 ]   C_D_0: C present, A not present, M=1
        //      [ X . 0 ]   C_D_0: C present, A not present, M=1, C a zombie
        //      [ . . 0 ]          only M=0 present, but nothing to do

        // Legend for action taken in the right half of the table:

        //      delete   live entry C(I(i),J(j)) marked for deletion (zombie)
        //      =A       live entry C(I(i),J(j)) is overwritten with new value
        //      =C+A     live entry C(I(i),J(j)) is modified with accum(c,a)
        //      C        live entry C(I(i),J(j)) is unchanged

        //      undelete entry C(I(i),J(j)) a zombie, bring back with A(i,j)
        //      X        entry C(I(i),J(j)) a zombie, no change, still zombie

        //      insert   entry C(I(i),J(j)) not present, add pending tuple
        //      .        entry C(I(i),J(j)) not present, no change

        //      blank    the table is left blank where the the event cannot
        //               occur:  GxB_subassign with no M cannot have
        //               M(i,j)=0, and GrB_setElement does not have the M
        //               column

        //----------------------------------------------------------------------
        // GrB_setElement and the Four Tables for GxB_subassign:
        //----------------------------------------------------------------------

            //------------------------------------------------------------
            // GrB_setElement:  no mask
            //------------------------------------------------------------

            // C A 1        =A                               |
            // X A 1        undelete                         |
            // . A 1        insert                           |

            //          GrB_setElement acts exactly like GxB_subassign with the
            //          implicit GrB_SECOND_Ctype operator, I=i, J=j, and a
            //          1-by-1 matrix A containing a single entry (not an
            //          implicit entry; there is no "." for A).  That is,
            //          nnz(A)==1.  No mask, and the descriptor is the default;
            //          C_replace effectively false, mask not complemented, A
            //          not transposed.  As a result, GrB_setElement can be
            //          freely mixed with calls to GxB_subassign with C_replace
            //          effectively false and with the identical
            //          GrB_SECOND_Ctype operator.  These calls to
            //          GxB_subassign can use the mask, either complemented or
            //          not, and they can transpose A if desired, and there is
            //          no restriction on I and J.  The matrix A can be any
            //          type and the type of A can change from call to call.

            //------------------------------------------------------------
            // NO accum  |  no mask     mask        mask
            // NO repl   |              not compl   compl
            //------------------------------------------------------------

            // C A 1        =A          =A          C        |
            // X A 1        undelete    undelete    X        |
            // . A 1        insert      insert      .        |

            // C . 1        delete      delete      C        |
            // X . 1        X           X           X        |
            // . . 1        .           .           .        |

            // C A 0                    C           =A       |
            // X A 0                    X           undelete |
            // . A 0                    .           insert   |

            // C . 0                    C           delete   |
            // X . 0                    X           X        |
            // . . 0                    .           .        |

            //          S_Extraction method works well: first extract pattern
            //          of S=C(I,J). Then examine all of A, M, S, and update
            //          C(I,J).  The method needs to examine all entries in
            //          in C(I,J) to delete them if A is not present, so
            //          S=C(I,J) is not costly.

            //------------------------------------------------------------
            // NO accum  |  no mask     mask        mask
            // WITH repl |              not compl   compl
            //------------------------------------------------------------

            // C A 1        =A          =A          delete   |
            // X A 1        undelete    undelete    X        |
            // . A 1        insert      insert      .        |

            // C . 1        delete      delete      delete   |
            // X . 1        X           X           X        |
            // . . 1        .           .           .        |

            // C A 0                    delete      =A       |
            // X A 0                    X           undelete |
            // . A 0                    .           insert   |

            // C . 0                    delete      delete   |
            // X . 0                    X           X        |
            // . . 0                    .           .        |

            //          S_Extraction method works well, since all of C(I,J)
            //          needs to be traversed, S=C(I,J) is reasonable to
            //          compute.

            //          With no accum: If there is no M and M is not
            //          complemented, then C_replace is irrelevant,  Whether
            //          true or false, the results in the two tables
            //          above are the same.

            //------------------------------------------------------------
            // ACCUM     |  no mask     mask        mask
            // NO repl   |              not compl   compl
            //------------------------------------------------------------

            // C A 1        =C+A        =C+A        C        |
            // X A 1        undelete    undelete    X        |
            // . A 1        insert      insert      .        |

            // C . 1        C           C           C        |
            // X . 1        X           X           X        |
            // . . 1        .           .           .        |

            // C A 0                    C           =C+A     |
            // X A 0                    X           undelete |
            // . A 0                    .           insert   |

            // C . 0                    C           C        |
            // X . 0                    X           X        |
            // . . 0                    .           .        |

            //          With ACCUM but NO C_replace: This method only needs to
            //          examine entries in A.  It does not need to examine all
            //          entries in C(I,J), nor all entries in M.  Entries in
            //          C but in not A remain unchanged.  This is like an
            //          extended GrB_setElement.  No entries in C can be
            //          deleted.  All other methods must examine all of C(I,J).

            //          Without S_Extraction: C(:,J) or M have many entries
            //          compared with A, do not extract S=C(I,J); use
            //          binary search instead.  Otherwise, use the same
            //          S_Extraction method as the other 3 cases.

            //          S_Extraction method: if nnz(C(:,j)) + nnz(M) is
            //          similar to nnz(A) then the S_Extraction method would
            //          work well.

            //------------------------------------------------------------
            // ACCUM     |  no mask     mask        mask
            // WITH repl |              not compl   compl
            //------------------------------------------------------------

            // C A 1        =C+A        =C+A        delete   |
            // X A 1        undelete    undelete    X        |
            // . A 1        insert      insert      .        |

            // C . 1        C           C           delete   |
            // X . 1        X           X           X        |
            // . . 1        .           .           .        |

            // C A 0                    delete      =C+A     |
            // X A 0                    X           undelete |
            // . A 0                    .           insert   |

            // C . 0                    delete      C        |
            // X . 0                    X           X        |
            // . . 0                    .           .        |

            //          S_Extraction method works well since all entries
            //          in C(I,J) must be examined.

            //          With accum: If there is no M and M is not
            //          complemented, then C_replace is irrelavant,  Whether
            //          true or false, the results in the two tables
            //          above are the same.

            //          This condition on C_replace holds with our without
            //          accum.  Thus, if there is no M, and M is
            //          not complemented, the C_replace can be set to false.

            //------------------------------------------------------------

            // ^^^^^ legend for left columns above:
            // C        prior entry C(I(i),J(j)) exists
            // X        prior entry C(I(i),J(j)) exists but is a zombie
            // .        no prior entry C(I(i),J(j))
            //   A      A(i,j) exists
            //   .      A(i,j) does not exist
            //     1    M(i,j)=1, assuming M exists (or if implicit)
            //     0    M(i,j)=0, only if M exists

        //----------------------------------------------------------------------
        // Actions in the Four Tables above
        //----------------------------------------------------------------------

            // Each entry in the Four Tables above are now explained in more
            // detail, describing what must be done in each case.  Zombies and
            // pending tuples are disjoint; they do not mix.  Zombies are IN
            // the pattern but pending tuples are updates that are NOT in the
            // pattern.  That is why a separate list of pending tuples must be
            // kept; there is no place for them in the pattern.  Zombies, on
            // the other hand, are entries IN the pattern that have been
            // marked for deletion.

            //--------------------------------
            // For entries NOT in the pattern:
            //--------------------------------

            // They can have pending tuples, and can acquire more.  No zombies.

            //      ( insert ):

            //          An entry C(I(i),J(j)) is NOT in the pattern, but its
            //          value must be modified.  This is an insertion, like
            //          GrB_setElement, and the insertion is added as a pending
            //          tuple for C(I(i),J(j)).  There can be many insertions
            //          to the same element, each in the list of pending
            //          tuples, in order of their insertion.  Eventually these
            //          pending tuples must be assembled into C(I(i),J(j)) in
            //          the right order using the implied SECOND operator.

            //      ( . ):

            //          no change.  C(I(i),J(j)) not in the pattern, and not
            //          modified.  This C(I(i),J(j)) position could have
            //          pending tuples, in the list of pending tuples, but none
            //          of them are changed.  If C_replace is true then those
            //          pending tuples would have to be discarded, but that
            //          condition will not occur because C_replace=true forces
            //          all prior tuples to the matrix to be assembled.

            //--------------------------------
            // For entries IN the pattern:
            //--------------------------------

            // They have no pending tuples, and acquire none.  It can be
            // zombie, can become a zombie, or a zombie can come back to life.

            //      ( delete ):

            //          C(I(i),J(j)) becomes a zombie, by flipping its row
            //          index via the GB_FLIP function.

            //      ( undelete ):

            //          C(I(i),J(j)) = A(i,j) was a zombie and is no longer a
            //          zombie.  Its row index is restored with GB_FLIP.

            //      ( X ):

            //          C(I(i),J(j)) was a zombie, and still is a zombie.
            //          row index is < 0, and actual index is GB_FLIP(I(i))

            //      ( C ):

            //          no change; C(I(i),J(j)) already an entry, and its value
            //          doesn't change.

            //      ( =A ):

            //          C(I(i),J(j)) = A(i,j), value gets overwritten.

            //      ( =C+A ):

            //          C(I(i),J(j)) = accum (C(I(i),J(j)), A(i,j))
            //          The existing balue is modified via the accumulator.


    //--------------------------------------------------------------------------
    // handling each action
    //--------------------------------------------------------------------------

        // Each of the 12 cases are handled by the following actions,
        // implemented as macros.  The Four Tables are re-sorted below,
        // and folded together according to their left column.

        // Once the M(i,j) entry is extracted, all GB_subassign_* functions
        // explicitly complement the scalar value if Mask_comp is true, before
        // using these action functions.  For the [no mask] case, M(i,j)=1.
        // Thus, only the middle column needs to be considered by each action;
        // the action will handle all three columns at the same time.  All
        // three columns remain in the re-sorted tables below for reference.

        //----------------------------------------------------------------------
        // ----[C A 1] or [X A 1]: C and A present, M=1
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------
            // C A 1        =A          =A          C        | no accum,no Crepl
            // C A 1        =A          =A          delete   | no accum,Crepl
            // C A 1        =C+A        =C+A        C        | accum, no Crepl
            // C A 1        =C+A        =C+A        delete   | accum, Crepl

            // X A 1        undelete    undelete    X        | no accum,no Crepl
            // X A 1        undelete    undelete    X        | no accum,Crepl
            // X A 1        undelete    undelete    X        | accum, no Crepl
            // X A 1        undelete    undelete    X        | accum, Crepl

            // Both C(I(i),J(j)) == S(i,j) and A(i,j) are present, and mij = 1.
            // C(I(i),J(i)) is updated with the entry A(i,j).
            // C_replace has no impact on this action.

            // [X A 1] matrix case
            #define GB_X_A_1_matrix                                         \
            {                                                               \
                /* ----[X A 1]                                           */ \
                /* action: ( undelete ): bring a zombie back to life     */ \
                GB_UNDELETE ;                                               \
                GB_COPY_aij_to_C (Cx,pC,Ax,pA,A_iso,cwork) ;                \
            }

            // [X A 1] scalar case
            #define GB_X_A_1_scalar                                         \
            {                                                               \
                /* ----[X A 1]                                           */ \
                /* action: ( undelete ): bring a zombie back to life     */ \
                GB_UNDELETE ;                                               \
                GB_COPY_scalar_to_C (Cx, pC, cwork) ;                       \
            }

            // [C A 1] matrix case when accum is present
            #define GB_withaccum_C_A_1_matrix                               \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 1]                                       */ \
                    /* action: ( undelete ): bring a zombie back to life */ \
                    GB_X_A_1_matrix ;                                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 1] with accum                            */ \
                    /* action: ( =C+A ): apply the accumulator           */ \
                    GB_ACCUMULATE_aij (Cx, pC, Ax, pA, A_iso, ywork) ;      \
                }                                                           \
            }

            // [C A 1] scalar case when accum is present
            #define GB_withaccum_C_A_1_scalar                               \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 1]                                       */ \
                    /* action: ( undelete ): bring a zombie back to life */ \
                    GB_X_A_1_scalar ;                                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 1] with accum, scalar expansion          */ \
                    /* action: ( =C+A ): apply the accumulator           */ \
                    GB_ACCUMULATE_scalar (Cx,pC,ywork) ;                    \
                }                                                           \
            }

            // [C A 1] matrix case when no accum is present
            #define GB_noaccum_C_A_1_matrix                                 \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 1]                                       */ \
                    /* action: ( undelete ): bring a zombie back to life */ \
                    GB_X_A_1_matrix ;                                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 1] no accum, scalar expansion            */ \
                    /* action: ( =A ): copy A into C                     */ \
                    GB_COPY_aij_to_C (Cx,pC,Ax,pA,A_iso,cwork) ;            \
                }                                                           \
            }

            // [C A 1] scalar case when no accum is present
            #define GB_noaccum_C_A_1_scalar                                 \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 1]                                       */ \
                    /* action: ( undelete ): bring a zombie back to life */ \
                    GB_X_A_1_scalar ;                                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 1] no accum, scalar expansion            */ \
                    /* action: ( =A ): copy A into C                     */ \
                    GB_COPY_scalar_to_C (Cx, pC, cwork) ;                   \
                }                                                           \
            }

        //----------------------------------------------------------------------
        // ----[. A 1]: C not present, A present, M=1
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------
            // . A 1        insert      insert      .        | no accum,no Crepl
            // . A 1        insert      insert      .        | no accum,Crepl
            // . A 1        insert      insert      .        | accum, no Crepl
            // . A 1        insert      insert      .        | accum, Crepl

            // C(I(i),J(j)) == S (i,j) is not present, A (i,j) is present, and
            // mij = 1. The mask M allows C to be written, but no entry present
            // in C (neither a live entry nor a zombie).  This entry must be
            // added to C but it doesn't fit in the pattern.  It is added as a
            // pending tuple.  Zombies and pending tuples do not intersect.

            // If adding the pending tuple fails, C is cleared entirely.
            // Otherwise the matrix C would be left in an incoherent partial
            // state of computation.  It's cleaner to just free it all.

            // The action is done by GB_PENDING_INSERT in GB_Pending.h.

            #if 0
            #define GB_D_A_1_scalar                                         \
            {                                                               \
                /* ----[. A 1]                                           */ \
                /* action: ( insert )                                    */ \
                GB_PENDING_INSERT (scalar) ;                                \
            }

            #define GB_D_A_1_matrix                                         \
            {                                                               \
                /* ----[. A 1]                                           */ \
                /* action: ( insert )                                    */ \
                GB_PENDING_INSERT_aij ;                                     \
            }
            #endif

        //----------------------------------------------------------------------
        // ----[C . 1] or [X . 1]: C present, A not present, M=1
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------
            // C . 1        delete      delete      C        | no accum,no Crepl
            // C . 1        delete      delete      delete   | no accum,Crepl
            // C . 1        C           C           C        | accum, no Crepl
            // C . 1        C           C           delete   | accum, Crepl

            // X . 1        X           X           X        | no accum,no Crepl
            // X . 1        X           X           X        | no accum,Crepl
            // X . 1        X           X           X        | accum, no Crepl
            // X . 1        X           X           X        | accum, Crepl

            // C(I(i),J(j)) == S (i,j) is present, A (i,j) not is present, and
            // mij = 1. The mask M allows C to be written, but no entry present
            // in A.  If no accum operator is present, C becomes a zombie.

            // This condition cannot occur if A is a dense matrix,
            // nor for scalar expansion

            // [C . 1] matrix case when no accum is present

            #if 0
            #define GB_noaccum_C_D_1_matrix                                 \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X . 1]                                       */ \
                    /* action: ( X ): still a zombie                     */ \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C . 1] no accum                              */ \
                    /* action: ( delete ): becomes a zombie              */ \
                    GB_DELETE ;                                             \
                }                                                           \
            }
            #endif

            // The above action is done via GB_DELETE_ENTRY.

        //----------------------------------------------------------------------
        // ----[C A 0] or [X A 0]: both C and A present but M=0
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------
            // C A 0                    C           =A       | no accum,no Crepl
            // C A 0                    delete      =A       | no accum,Crepl
            // C A 0                    C           =C+A     | accum, no Crepl
            // C A 0                    delete      =C+A     | accum, Crepl

            // X A 0                    X           undelete | no accum,no Crepl
            // X A 0                    X           undelete | no accum,Crepl
            // X A 0                    X           undelete | accum, no Crepl
            // X A 0                    X           undelete | accum, Crepl

            // Both C(I(i),J(j)) == S(i,j) and A(i,j) are present, and mij = 0.
            // The mask prevents A being written to C, so A has no effect on
            // the result.  If C_replace is true, however, the entry is
            // deleted, becoming a zombie.  This case does not occur if
            // the mask M is not present.  This action also handles the
            // [C . 0] and [X . 0] cases; see the next section below.

            // This condition can still occur if A is dense, so if a mask M is
            // present, entries can still be deleted from C.  As a result, the
            // fact that A is dense cannot be exploited when the mask M is
            // present.

            #if 0
            #define GB_C_A_0                                                \
            {                                                               \
                if (is_zombie)                                              \
                {                                                           \
                    /* ----[X A 0]                                       */ \
                    /* ----[X . 0]                                       */ \
                    /* action: ( X ): still a zombie                     */ \
                }                                                           \
                else if (C_replace)                                         \
                {                                                           \
                    /* ----[C A 0] replace                               */ \
                    /* ----[C . 0] replace                               */ \
                    /* action: ( delete ): becomes a zombie              */ \
                    GB_DELETE ;                                             \
                }                                                           \
                else                                                        \
                {                                                           \
                    /* ----[C A 0] no replace                            */ \
                    /* ----[C . 0] no replace                            */ \
                    /* action: ( C ): no change                          */ \
                }                                                           \
            }
            #endif

            // The above action is done via GB_DELETE_ENTRY.

            // The above action is very similar to C_D_1.  The only difference
            // is how the entry C becomes a zombie.  With C_D_1, there is no
            // entry in A, so C becomes a zombie if no accum function is used
            // because the implicit value A(i,j) gets copied into C, causing it
            // to become an implicit value also (deleting the entry in C).
            // With C_A_0, the entry C is protected from any modification from
            // A (regardless of accum or not).  However, if C_replace is true,
            // the entry is cleared.  The mask M does not protect C from the
            // C_replace action.

            // If C_replace is false, then the [C A 0] action does nothing.
            // If C_replace is true, then the action becomes the following:

            #define GB_DELETE_ENTRY                                         \
            {                                                               \
                if (!is_zombie)                                             \
                {                                                           \
                    GB_DELETE ;                                             \
                }                                                           \
            }

        //----------------------------------------------------------------------
        // ----[C . 0] or [X . 0]: C present, A not present, M=0
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------

            // C . 0                    C           delete   | no accum,no Crepl
            // C . 0                    delete      delete   | no accum,Crepl
            // C . 0                    C           C        | accum, no Crepl
            // C . 0                    delete      C        | accum, Crepl

            // X . 0                    X           X        | no accum,no Crepl
            // X . 0                    X           X        | no accum,Crepl
            // X . 0                    X           X        | accum, no Crepl
            // X . 0                    X           X        | accum, Crepl

            // C(I(i),J(j)) == S(i,j) is present, but A(i,j) is not present,
            // and mij = 0.  Since A(i,j) has no effect on the result,
            // this is the same as the C_A_0 action above.

            // This condition cannot occur if A is a dense matrix, nor for
            // scalar expansion, but the existance of the entry A is not
            // relevant.

            // If C_replace is false, then the [C D 0] action does nothing.
            // If C_replace is true, then the action becomes GB_DELETE_ENTRY.

            #if 0
            #define GB_C_D_0 GB_C_A_0
            #endif

        //----------------------------------------------------------------------
        // ----[. A 0]: C not present, A present, M=0
        //----------------------------------------------------------------------

            // . A 0                    .           insert   | no accum,no Crepl
            // . A 0                    .           insert   | no accum,no Crepl
            // . A 0                    .           insert   | accum, no Crepl
            // . A 0                    .           insert   | accum, Crepl

            // C(I(i),J(j)) == S(i,j) is not present, A(i,j) is present,
            // but mij = 0.  The mask M prevents A from modifying C, so the
            // A(i,j) entry is ignored.  C_replace has no effect since the
            // entry is already cleared.  There is nothing to do.

        //----------------------------------------------------------------------
        // ----[. . 1] and [. . 0]: no entries in C and A, M = 0 or 1
        //----------------------------------------------------------------------

            //------------------------------------------------
            //           |  no mask     mask        mask
            //           |              not compl   compl
            //------------------------------------------------

            // . . 1        .           .           .        | no accum,no Crepl
            // . . 1        .           .           .        | no accum,Crepl
            // . . 1        .           .           .        | accum, no Crepl
            // . . 1        .           .           .        | accum, Crepl

            // . . 0        .           .           .        | no accum,no Crepl
            // . . 0        .           .           .        | no accum,Crepl
            // . . 0        .           .           .        | accum, no Crepl
            // . . 0        .           .           .        | accum, Crepl

            // Neither C(I(i),J(j)) == S(i,j) nor A(i,j) are not present,
            // Nothing happens.  The M(i,j) entry is present, otherwise
            // this (i,j) position would not be considered at all.
            // The M(i,j) entry has no effect.  There is nothing to do.

//------------------------------------------------------------------------------
// GB_ALLOCATE_NPENDING_WERK: allocate Npending workspace
//------------------------------------------------------------------------------

#define GB_ALLOCATE_NPENDING_WERK                                           \
    GB_WERK_PUSH (Npending, ntasks+1, int64_t) ;                            \
    if (Npending == NULL)                                                   \
    {                                                                       \
        GB_FREE_ALL ;                                                       \
        return (GrB_OUT_OF_MEMORY) ;                                        \
    }

//------------------------------------------------------------------------------
// GB_SUBASSIGN_ONE_SLICE: slice one matrix (M)
//------------------------------------------------------------------------------

// Methods: 05, 06n, 07.  If C is dense, it is sliced for a fine task, so that
// it can do a binary search via GB_iC_BINARY_SEARCH.  But if C(:,jC) is dense,
// C(:,jC) is not sliced, so the fine task must do a direct lookup via
// GB_iC_DENSE_LOOKUP.  Otherwise a race condition will occur.

#define GB_SUBASSIGN_ONE_SLICE(M)                                           \
    GB_OK (GB_subassign_one_slice (                                         \
        &TaskList, &TaskList_size, &ntasks, &nthreads,                      \
        C, I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,                      \
        M, Werk)) ;                                                         \
    GB_ALLOCATE_NPENDING_WERK ;

//------------------------------------------------------------------------------
// GB_SUBASSIGN_TWO_SLICE: slice two matrices
//------------------------------------------------------------------------------

// Methods: 02, 04, 06s_and_14, 08s_and_16, 09, 10_and_18, 11, 12_and_20

// Create tasks for Z = X+S, and the mapping of Z to X and S.  The matrix X is
// either A or M.  No need to examine C, since it will be accessed via S, not
// via binary search.

// If X is bitmap, this method is not used.  Instead, GB_SUBASSIGN_IXJ_SLICE is
// used to iterate over the matrix X.

#define GB_SUBASSIGN_TWO_SLICE(X,S)                                         \
    int Z_sparsity = GxB_SPARSE ;                                           \
    int64_t Znvec ;                                                         \
    GB_OK (GB_add_phase0 (                                                  \
        &Znvec, &Zh, &Zh_size, NULL, NULL, &Z_to_X, &Z_to_X_size,           \
        &Z_to_S, &Z_to_S_size, NULL, &Z_sparsity,                           \
        NULL, X, S, Werk)) ;                                                \
    GB_OK (GB_ewise_slice (                                                 \
        &TaskList, &TaskList_size, &ntasks, &nthreads,                      \
        Znvec, Zh, NULL, Z_to_X, Z_to_S, false,                             \
        NULL, X, S, Werk)) ;                                                \
    GB_ALLOCATE_NPENDING_WERK ;

//------------------------------------------------------------------------------
// GB_SUBASSIGN_IXJ_SLICE: slice IxJ for a scalar assignement method
//------------------------------------------------------------------------------

// Methods: 01, 03, 13, 15, 17, 19.  All of these methods access the C matrix
// via S, not via binary search.

#define GB_SUBASSIGN_IXJ_SLICE                                              \
    GB_OK (GB_subassign_IxJ_slice (                                         \
        &TaskList, &TaskList_size, &ntasks, &nthreads,                      \
        /* I, */ nI, /* Ikind, Icolon, J, */ nJ, /* Jkind, Jcolon, */       \
        Werk)) ;                                                            \
    GB_ALLOCATE_NPENDING_WERK ;

//------------------------------------------------------------------------------
// GB_GET_TASK_DESCRIPTOR: get coarse/fine task descriptor
//------------------------------------------------------------------------------

#define GB_GET_TASK_DESCRIPTOR                                              \
    int64_t kfirst = TaskList [taskid].kfirst ;                             \
    int64_t klast  = TaskList [taskid].klast ;                              \
    bool fine_task = (klast == -1) ;                                        \
    if (fine_task)                                                          \
    {                                                                       \
        /* a fine task operates on a slice of a single vector */            \
        klast = kfirst ;                                                    \
    }                                                                       \

#define GB_GET_TASK_DESCRIPTOR_PHASE1                                       \
    GB_GET_TASK_DESCRIPTOR ;                                                \
    int64_t task_nzombies = 0 ;                                             \
    int64_t task_pending = 0 ;

//------------------------------------------------------------------------------
// GB_GET_MAPPED: get the content of a vector for a coarse/fine task
//------------------------------------------------------------------------------

#define GB_GET_MAPPED(pX_start, pX_fini, pX, pX_end, Xp, j, k, Z_to_X, Xvlen) \
    int64_t pX_start = -1, pX_fini = -1 ;                                   \
    if (fine_task)                                                          \
    {                                                                       \
        /* A fine task operates on a slice of X(:,k) */                     \
        pX_start = TaskList [taskid].pX ;                                   \
        pX_fini  = TaskList [taskid].pX_end ;                               \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* vectors are never sliced for a coarse task */                    \
        int64_t kX = (Z_to_X == NULL) ? j : Z_to_X [k] ;                    \
        if (kX >= 0)                                                        \
        {                                                                   \
            pX_start = GBP (Xp, kX, Xvlen) ;                                \
            pX_fini  = GBP (Xp, kX+1, Xvlen) ;                              \
        }                                                                   \
    }

//------------------------------------------------------------------------------
// GB_GET_EVEC: get the content of a vector for Method08n
//------------------------------------------------------------------------------

#define GB_GET_EVEC(pX_start, pX_fini, pX, pX_end, Xp, Xh, j,k,Z_to_X,Xvlen)\
    int64_t pX_start = -1, pX_fini = -1 ;                                   \
    if (fine_task)                                                          \
    {                                                                       \
        /* A fine task operates on a slice of X(:,k) */                     \
        pX_start = TaskList [taskid].pX ;                                   \
        pX_fini  = TaskList [taskid].pX_end ;                               \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* vectors are never sliced for a coarse task */                    \
        int64_t kX = (Zh_shallow == Xh) ? k :                               \
            ((Z_to_X == NULL) ? j : Z_to_X [k]) ;                           \
        if (kX >= 0)                                                        \
        {                                                                   \
            pX_start = GBP (Xp, kX, Xvlen) ;                                \
            pX_fini  = GBP (Xp, kX+1, Xvlen) ;                              \
        }                                                                   \
    }

//------------------------------------------------------------------------------
// GB_GET_IXJ_TASK_DESCRIPTOR*: get the task descriptor for IxJ
//------------------------------------------------------------------------------

// Q denotes the Cartesian product IXJ

#define GB_GET_IXJ_TASK_DESCRIPTOR(iQ_start,iQ_end)                         \
    GB_GET_TASK_DESCRIPTOR ;                                                \
    int64_t iQ_start = 0, iQ_end = nI ;                                     \
    if (fine_task)                                                          \
    {                                                                       \
        iQ_start = TaskList [taskid].pA ;                                   \
        iQ_end   = TaskList [taskid].pA_end ;                               \
    }

#define GB_GET_IXJ_TASK_DESCRIPTOR_PHASE1(iQ_start,iQ_end)                  \
    GB_GET_IXJ_TASK_DESCRIPTOR (iQ_start, iQ_end)                           \
    int64_t task_nzombies = 0 ;                                             \
    int64_t task_pending = 0 ;

#define GB_GET_IXJ_TASK_DESCRIPTOR_PHASE2(iQ_start,iQ_end)                  \
    GB_GET_IXJ_TASK_DESCRIPTOR (iQ_start, iQ_end)                           \
    GB_START_PENDING_INSERTION ;

//--------------------------------------------------------------------------
// GB_LOOKUP_VECTOR
//--------------------------------------------------------------------------

// Find pX_start and pX_end for the vector X (:,j)

#define GB_LOOKUP_VECTOR(pX_start,pX_end,X,j)                           \
{                                                                       \
    if (X ## _is_hyper)                                                 \
    {                                                                   \
        GB_hyper_hash_lookup (X ## h, X ## nvec, X ## p, X ## _Yp,      \
            X ## _Yi, X ## _Yx, X ## _hash_bits, j, &pX_start, &pX_end) ;   \
    }                                                                   \
    else                                                                \
    {                                                                   \
        pX_start = GBP (X ## p, j  , X ## vlen) ;                       \
        pX_end   = GBP (X ## p, j+1, X ## vlen) ;                       \
    }                                                                   \
}

//------------------------------------------------------------------------------
// GB_LOOKUP_VECTOR_jC: get the vector C(:,jC)
//------------------------------------------------------------------------------

#define GB_LOOKUP_VECTOR_jC(fine_task,taskid)                               \
    /* lookup jC in C */                                                    \
    /* jC = J [j] ; or J is ":" or jbegin:jend or jbegin:jinc:jend */       \
    int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;                          \
    int64_t pC_start, pC_end ;                                              \
    if (fine_task)                                                          \
    {                                                                       \
        pC_start = TaskList [taskid].pC ;                                   \
        pC_end   = TaskList [taskid].pC_end ;                               \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        GB_LOOKUP_VECTOR (pC_start, pC_end, C, jC) ;                        \
    }

//------------------------------------------------------------------------------
// GB_LOOKUP_VECTOR_FOR_IXJ: get the start of a vector for scalar assignment
//------------------------------------------------------------------------------

// Find pX and pX_end for the vector X (iQ_start:iQ_end, j), for a scalar
// assignment method, or a method iterating over all IxJ for a bitmap M or A.

#define GB_LOOKUP_VECTOR_FOR_IXJ(X,iQ_start)                                \
    int64_t p ## X, p ## X ## _end ;                                        \
    GB_LOOKUP_VECTOR (p ## X, p ## X ## _end, X, j) ;                       \
    if (iQ_start != 0)                                                      \
    {                                                                       \
        if (X ## i == NULL)                                                 \
        {                                                                   \
            /* X is full or bitmap */                                       \
            p ## X += iQ_start ;                                            \
        }                                                                   \
        else                                                                \
        {                                                                   \
            /* X is sparse or hypersparse */                                \
            int64_t pright = p ## X ## _end - 1 ;                           \
            bool found ;                                                    \
            GB_SPLIT_BINARY_SEARCH (iQ_start, X ## i, p ## X, pright, found) ;\
        }                                                                   \
    }

//------------------------------------------------------------------------------
// GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP
//------------------------------------------------------------------------------

// mij = M(i,j)

#define GB_MIJ_BINARY_SEARCH_OR_DENSE_LOOKUP(i)                             \
    bool mij ;                                                              \
    if (M_is_bitmap)                                                        \
    {                                                                       \
        /* M(:,j) is bitmap, no need for binary search */                   \
        int64_t pM = pM_start + i ;                                         \
        mij = Mb [pM] && GB_MCAST (Mx, pM, msize) ;                         \
    }                                                                       \
    else if (mjdense)                                                       \
    {                                                                       \
        /* M(:,j) is dense, no need for binary search */                    \
        int64_t pM = pM_start + i ;                                         \
        mij = GB_MCAST (Mx, pM, msize) ;                                    \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* M(:,j) is sparse, binary search for M(i,j) */                    \
        int64_t pM     = pM_start ;                                         \
        int64_t pright = pM_end - 1 ;                                       \
        bool found ;                                                        \
        GB_BINARY_SEARCH (i, Mi, pM, pright, found) ;                       \
        if (found)                                                          \
        {                                                                   \
            mij = GB_MCAST (Mx, pM, msize) ;                                \
        }                                                                   \
        else                                                                \
        {                                                                   \
            mij = false ;                                                   \
        }                                                                   \
    }                                                                       \

//------------------------------------------------------------------------------
// GB_PHASE1_TASK_WRAPUP: wrapup for a task in phase 1
//------------------------------------------------------------------------------

// sum up the zombie count, and record the # of pending tuples for this task

#define GB_PHASE1_TASK_WRAPUP                                               \
    nzombies += task_nzombies ;                                             \
    Npending [taskid] = task_pending ;

//------------------------------------------------------------------------------
// GB_PENDING_CUMSUM: finalize zombies, count # pending tuples for all tasks
//------------------------------------------------------------------------------

#define GB_PENDING_CUMSUM                                                   \
    C->nzombies = nzombies ;                                                \
    GB_cumsum (Npending, ntasks, NULL, 1, NULL) ;                           \
    int64_t nnew = Npending [ntasks] ;                                      \
    if (nnew == 0)                                                          \
    {                                                                       \
        /* no pending tuples, so skip phase 2 */                            \
        GB_FREE_ALL ;                                                       \
        ASSERT_MATRIX_OK (C, "C, no pending tuples ", GB_FLIP (GB0)) ;      \
        return (GrB_SUCCESS) ;                                              \
    }                                                                       \
    /* ensure that C->Pending is large enough to handle nnew more tuples */ \
    if (!GB_Pending_ensure (&(C->Pending), C_iso, atype, accum, is_matrix,  \
        nnew, Werk))                                                        \
    {                                                                       \
        GB_FREE_ALL ;                                                       \
        return (GrB_OUT_OF_MEMORY) ;                                        \
    }                                                                       \
    GB_Pending Pending = C->Pending ;                                       \
    int64_t *restrict Pending_i = Pending->i ;                              \
    int64_t *restrict Pending_j = Pending->j ;                              \
    GB_void *restrict Pending_x = Pending->x ; /* NULL if C is iso */       \
    int64_t npending_orig = Pending->n ;                                    \
    bool pending_sorted = Pending->sorted ;

//------------------------------------------------------------------------------
// GB_START_PENDING_INSERTION: start insertion of pending tuples (phase 2)
//------------------------------------------------------------------------------

#define GB_START_PENDING_INSERTION                                          \
    bool task_sorted = true ;                                               \
    int64_t ilast = -1 ;                                                    \
    int64_t jlast = -1 ;                                                    \
    int64_t n = Npending [taskid] ;                                         \
    int64_t task_pending = Npending [taskid+1] - n ;                        \
    if (task_pending == 0) continue ;                                       \
    n += npending_orig ;

#define GB_GET_TASK_DESCRIPTOR_PHASE2                                       \
    GB_GET_TASK_DESCRIPTOR ;                                                \
    GB_START_PENDING_INSERTION ;

//------------------------------------------------------------------------------
// GB_PENDING_INSERT: add (iC,jC,aij) or just (iC,aij) if Pending_j is NULL
//------------------------------------------------------------------------------

// GB_PENDING_INSERT(aij) is used by GB_subassign_* to insert a pending tuple,
// in phase 2.  The list has already been reallocated after phase 1 to hold all
// the new pending tuples, so GB_Pending_realloc is not required.  If C is iso,
// Pending->x is NULL.

#define GB_PENDING_INSERT(aij)                                              \
    if (task_sorted)                                                        \
    {                                                                       \
        if (!((jlast < jC) || (jlast == jC && ilast <= iC)))                \
        {                                                                   \
            task_sorted = false ;                                           \
        }                                                                   \
    }                                                                       \
    Pending_i [n] = iC ;                                                    \
    if (Pending_j != NULL) Pending_j [n] = jC ;                             \
    if (Pending_x != NULL) memcpy (Pending_x +(n*asize), (aij), asize) ;    \
    n++ ;                                                                   \
    ilast = iC ;                                                            \
    jlast = jC ;

// insert A(i,j) into the list of pending tuples
#define GB_PENDING_INSERT_aij                                               \
    GB_PENDING_INSERT (Ax + (A_iso ? 0 : ((pA)*asize)))

//------------------------------------------------------------------------------
// GB_PHASE2_TASK_WRAPUP: wrapup for a task in phase 2
//------------------------------------------------------------------------------

#define GB_PHASE2_TASK_WRAPUP                                               \
    pending_sorted = pending_sorted && task_sorted ;                        \
    ASSERT (n == npending_orig + Npending [taskid+1]) ;

//------------------------------------------------------------------------------
// GB_SUBASSIGN_WRAPUP: finalize the subassign method after phase 2
//------------------------------------------------------------------------------

// If pending_sorted is true, then the original pending tuples (if any) were
// sorted, and each task found that its tuples were also sorted.  The
// boundaries between each task must now be checked.

#define GB_SUBASSIGN_WRAPUP                                                 \
    if (pending_sorted)                                                     \
    {                                                                       \
        for (int taskid = 0 ; pending_sorted && taskid < ntasks ; taskid++) \
        {                                                                   \
            int64_t n = Npending [taskid] ;                                 \
            int64_t task_pending = Npending [taskid+1] - n ;                \
            n += npending_orig ;                                            \
            if (task_pending > 0 && n > 0)                                  \
            {                                                               \
                /* (i,j) is the first pending tuple for this task; check */ \
                /* with the pending tuple just before it                 */ \
                ASSERT (n < npending_orig + nnew) ;                         \
                int64_t i = Pending_i [n] ;                                 \
                int64_t j = (Pending_j != NULL) ? Pending_j [n] : 0 ;       \
                int64_t ilast = Pending_i [n-1] ;                           \
                int64_t jlast = (Pending_j != NULL) ? Pending_j [n-1] : 0 ; \
                pending_sorted = pending_sorted &&                          \
                    ((jlast < j) || (jlast == j && ilast <= i)) ;           \
            }                                                               \
        }                                                                   \
    }                                                                       \
    Pending->n += nnew ;                                                    \
    Pending->sorted = pending_sorted ;                                      \
    GB_FREE_ALL ;                                                           \
    ASSERT_MATRIX_OK (C, "C with pending tuples", GB_FLIP (GB0)) ;          \
    return (GrB_SUCCESS) ;

//==============================================================================
// macros for bitmap_assign methods
//==============================================================================

//------------------------------------------------------------------------------
// burble
//------------------------------------------------------------------------------

#define GBURBLE_BITMAP_ASSIGN(method,M,Mask_comp,accum,Ikind,Jkind,akind)   \
    GBURBLE ("Method:" method " ") ;                                        \
    GB_assign_burble (C_replace, Ikind, Jkind, M, Mask_comp,                \
        Mask_struct, accum, A, akind) ;

//------------------------------------------------------------------------------
// GB_GET_C_BITMAP: get the C matrix (must be bitmap)
//------------------------------------------------------------------------------

// C must be a bitmap matrix

#define GB_GET_C_BITMAP                                                     \
    GrB_Info info ;                                                         \
    /* also get the max # of threads to use */                              \
    int nthreads_max = GB_Context_nthreads_max ( ) ;                        \
    double chunk = GB_Context_chunk ( ) ;                                   \
    ASSERT_MATRIX_OK (C, "C for bitmap assign", GB0) ;                      \
    ASSERT (GB_IS_BITMAP (C)) ;                                             \
    int8_t  *Cb = C->b ;                                                    \
    const bool C_iso = C->iso ;                                             \
    GB_void *Cx = (C_iso) ? NULL : (GB_void *) C->x ;                       \
    const size_t csize = C->type->size ;                                    \
    const GB_Type_code ccode = C->type->code ;                              \
    const int64_t cvdim = C->vdim ;                                         \
    const int64_t cvlen = C->vlen ;                                         \
    const int64_t vlen = cvlen ;    /* for GB_bitmap_assign_IxJ_template */ \
    const int64_t cnzmax = cvlen * cvdim ;                                  \
    int64_t cnvals = C->nvals ;

//------------------------------------------------------------------------------
// GB_SLICE_M: slice the mask matrix M
//------------------------------------------------------------------------------

#define GB_SLICE_M                                                          \
    GB_GET_M                                                                \
    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;                               \
    int M_ntasks, M_nthreads ;                                              \
    GB_M_NHELD (M_nnz_held) ;                                               \
    GB_SLICE_MATRIX_WORK (M, 8, M_nnz_held + M->nvec, M_nnz_held) ;

//------------------------------------------------------------------------------
// GB_GET_A_AND_SCALAR: get the A matrix or the scalar
//------------------------------------------------------------------------------

// ALIAS of C and A for bitmap methods: OK only for C(:,:)=A assignment.

#define GB_GET_A_AND_SCALAR                                                 \
    const int64_t *Ap = NULL ;                                              \
    const int64_t *Ah = NULL ;                                              \
    const int8_t  *Ab = NULL ;                                              \
    const int64_t *Ai = NULL ;                                              \
    const GB_void *Ax = NULL ;                                              \
    const bool A_iso = (A == NULL) ? false : A->iso ;                       \
    const GrB_Type atype = (A == NULL) ? scalar_type : A->type ;            \
    const size_t       asize = atype->size ;                                \
    const GB_Type_code acode = atype->code ;                                \
    if (A != NULL)                                                          \
    {                                                                       \
        ASSERT_MATRIX_OK (A, "A for bitmap assign/subassign", GB0) ;        \
        Ap = A->p ;                                                         \
        Ah = A->h ;                                                         \
        Ab = A->b ;                                                         \
        Ai = A->i ;                                                         \
        Ax = (C_iso) ? NULL : (GB_void *) A->x ;                            \
    }                                                                       \
    GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;         \
    GB_DECLAREC (cwork) ;                                                   \
    if (!C_iso)                                                             \
    {                                                                       \
        if (A == NULL)                                                      \
        {                                                                   \
            /* cwork = (ctype) scalar */                                    \
            cast_A_to_C (cwork, scalar, asize) ;                            \
        }                                                                   \
        else if (A_iso)                                                     \
        {                                                                   \
            /* cwork = (ctype) Ax [0], typecast iso value of A into cwork */\
            cast_A_to_C (cwork, Ax, asize) ;                                \
        }                                                                   \
    }

//------------------------------------------------------------------------------
// GB_GET_ACCUM_FOR_BITMAP: get accum op and its related typecasting functions
//------------------------------------------------------------------------------

#define GB_GET_ACCUM_FOR_BITMAP                                             \
    GB_GET_ACCUM ;                                                          \
    GB_DECLAREY (ywork) ;                                                   \
    if (!C_iso)                                                             \
    {                                                                       \
        if (A == NULL)                                                      \
        {                                                                   \
            cast_A_to_Y (ywork, scalar, asize) ;                            \
        }                                                                   \
        else if (A_iso)                                                     \
        {                                                                   \
            cast_A_to_Y (ywork, Ax, asize) ;                                \
        }                                                                   \
    }

#endif

