//------------------------------------------------------------------------------
// GB_select_bitmap:  select entries from a bitmap or full matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

#include "GB_select.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_sel__include.h"
#endif

#define GB_FREE_ALL         \
    GB_phybix_free (C) ;

GrB_Info GB_select_bitmap
(
    GrB_Matrix C,               // output matrix, static header
    const bool C_iso,           // if true, C is iso
    const GrB_IndexUnaryOp op,
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    const int64_t ithunk,       // (int64_t) Thunk, if Thunk is NULL
    const GB_void *restrict athunk,     // (A->type) Thunk
    const GB_void *restrict ythunk,     // (op->ytype) Thunk
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A for bitmap selector", GB0) ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for bitmap selector", GB0) ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    GB_Opcode opcode = op->opcode ;
    ASSERT (opcode != GB_NONZOMBIE_idxunop_code) ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz_held (A) ;
    const size_t asize = A->type->size ;
    const GB_Type_code acode = A->type->code ;

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    // C->b and C->x are malloc'd, not calloc'd
    // set C->iso = C_iso   OK
    GB_OK (GB_new_bix (&C, // always bitmap, existing header
        A->type, A->vlen, A->vdim, GB_Ap_calloc, true,
        GxB_BITMAP, false, A->hyper_switch, -1, anz, true, C_iso)) ;
    int64_t cnvals ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // copy values of A into C
    //--------------------------------------------------------------------------

    // C and A have the same type, since C is the T matrix from GB_select,
    // not the final user's C matrix.  In the future A could be typecasted
    // into C in the JIT kernel.

    if (C_iso)
    { 
        // Cx [0] = Ax [0] or (A->type) thunk
        GB_select_iso (C->x, opcode, athunk, A->x, asize) ;
    }
    else
    { 
        // Cx [0:anz-1] = Ax [0:anz-1]
        GB_memcpy (C->x, A->x, anz * asize, nthreads) ;
    }

    //--------------------------------------------------------------------------
    // bitmap selector kernel
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;

    if (GB_IS_INDEXUNARYOP_CODE_POSITIONAL (opcode))
    { 

        //----------------------------------------------------------------------
        // bitmap selector for positional ops
        //----------------------------------------------------------------------

        info = GB_select_positional_bitmap (C->b, &cnvals, A, ithunk, op,
            nthreads) ;
    }
    else
    { 

        //----------------------------------------------------------------------
        // bitmap selector for VALUE* and user-defined ops
        //----------------------------------------------------------------------

        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        { 

            //------------------------------------------------------------------
            // via the factory kernel 
            //------------------------------------------------------------------

            #define GB_selbit(opname,aname) GB (_sel_bitmap_ ## opname ## aname)
            #define GB_SEL_WORKER(opname,aname)                         \
            {                                                           \
                info = GB_selbit (opname, aname) (C->b, &cnvals, A,     \
                    ythunk, nthreads) ;                                 \
            }                                                           \
            break ;

            #include "GB_select_entry_factory.c"
        }
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_select_bitmap_jit (C->b, &cnvals, C_iso,
                A, flipij, ythunk, op, nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel 
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_select_generic_bitmap (C->b, &cnvals, A, flipij, ythunk,
                op, nthreads) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    C->nvals = cnvals ;
    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C from bitmap selector", GB0) ;
    return (GrB_SUCCESS) ;
}

