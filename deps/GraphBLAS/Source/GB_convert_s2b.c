//------------------------------------------------------------------------------
// GB_convert_s2b: convert from sparse/hypersparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// The matrix A is converted from sparse/hypersparse to bitmap.
// FUTURE: A could also be typecasted and/or a unary operator applied,
// via the JIT kernel.

#include "GB_apply.h"
#include "GB_ek_slice.h"
#include "GB_stringify.h"

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;   \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    GB_FREE (&Ax_new, Ax_size) ;            \
    GB_FREE (&Ab, Ab_size) ;                \
}

GrB_Info GB_convert_s2b    // convert sparse/hypersparse to bitmap
(
    GrB_Matrix A,               // matrix to convert from sparse to bitmap
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    int8_t  *restrict Ab      = NULL ; size_t Ab_size = 0 ;
    GB_void *restrict Ax_new  = NULL ; size_t Ax_size = 0 ;
    GB_void *restrict Ax_keep = NULL ;

    ASSERT_MATRIX_OK (A, "A converting sparse/hypersparse to bitmap", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;        // A can be jumbled on input
    ASSERT (GB_ZOMBIES_OK (A)) ;        // A can have zombies on input

    //--------------------------------------------------------------------------
    // determine the maximum number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // determine if the conversion can be done in-place
    //--------------------------------------------------------------------------

    // A->x does not change if A is as-if-full or A is iso
    bool A_iso = A->iso ;
    bool A_as_if_full = GB_as_if_full (A) ;
    bool in_place = A_as_if_full || A_iso ;

    //--------------------------------------------------------------------------
    // allocate A->b
    //--------------------------------------------------------------------------

    const int64_t anz = GB_nnz (A) ;
    GB_BURBLE_N (anz, "(sparse to bitmap) ") ;
    const int64_t avdim = A->vdim ;
    const int64_t avlen = A->vlen ;
    int64_t anzmax ;
    if (!GB_int64_multiply ((GrB_Index *) (&anzmax), avdim, avlen))
    { 
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }
    anzmax = GB_IMAX (anzmax, 1) ;
    Ab = GB_MALLOC (anzmax, int8_t, &Ab_size) ;
    if (Ab == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // allocate the new A->x
    //--------------------------------------------------------------------------

    const size_t asize = A->type->size ;
    bool Ax_shallow ;

    if (in_place)
    { 
        // keep the existing A->x
        Ax_keep = (GB_void *) A->x ;
        Ax_shallow = A->x_shallow ; Ax_size = A->x_size ;
    }
    else
    {
        // A->x must be modified to fit the bitmap structure.  A->x is calloc'd
        // since otherwise it would contain uninitialized values where A->b is
        // false and entries are not present.
        Ax_new = GB_CALLOC (anzmax * asize, GB_void, &Ax_size) ; // x:OK:calloc
        Ax_shallow = false ;
        if (Ax_new == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        Ax_keep = Ax_new ;
    }

    //--------------------------------------------------------------------------
    // scatter the pattern and values into the new bitmap
    //--------------------------------------------------------------------------

    int64_t nzombies = A->nzombies ;
    if (A_as_if_full)
    { 

        //----------------------------------------------------------------------
        // the sparse A has all entries or is iso: convert in-place
        //----------------------------------------------------------------------

        ASSERT (nzombies == 0) ;
        // set all of Ab [0..anz-1] to 1, in parallel
        GB_memset (Ab, 1, anz, nthreads_max) ;
        info = GrB_SUCCESS ;

    }
    else
    {

        //----------------------------------------------------------------------
        // set all of Ab to zero
        //----------------------------------------------------------------------

        GB_memset (Ab, 0, anzmax, nthreads_max) ;

        //----------------------------------------------------------------------
        // scatter the values and pattern of A into the bitmap
        //----------------------------------------------------------------------

        int A_nthreads, A_ntasks ;
        GB_SLICE_MATRIX (A, 8) ;

        info = GrB_NO_VALUE ;

        if (A_iso)
        { 
            // A is iso; numerical entries are not modified
            #undef  GB_COPY
            #define GB_COPY(Axnew,pnew,Ax,p) ;
            #include "GB_convert_s2b_template.c"
            info = GrB_SUCCESS ;
        }
        else
        {

            #ifndef GBCOMPACT
            GB_IF_FACTORY_KERNELS_ENABLED
            { 
                switch (asize)
                {
                    #undef  GB_COPY
                    #define GB_COPY(Axnew,pnew,Ax,p)         \
                        Axnew [pnew] = Ax [p] ;

                    case GB_1BYTE : // uint8, int8, bool, or 1-byte user
                        #define GB_A_TYPE uint8_t
                        #include "GB_convert_s2b_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    case GB_2BYTE : // uint16, int16, or 2-byte user-defined
                        #define GB_A_TYPE uint16_t
                        #include "GB_convert_s2b_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    case GB_4BYTE : // uint32, int32, float, or 4-byte user
                        #define GB_A_TYPE uint32_t
                        #include "GB_convert_s2b_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    case GB_8BYTE : // uint64, int64, double, float complex,
                             // or 8-byte user defined
                        #define GB_A_TYPE uint64_t
                        #include "GB_convert_s2b_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    case GB_16BYTE : // double complex or 16-byte user-defined
                        #define GB_A_TYPE GB_blob16
                        #include "GB_convert_s2b_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    default:;
                }
            }
            #endif

            //------------------------------------------------------------------
            // via the JIT or PreJIT kernel
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            { 
                struct GB_UnaryOp_opaque op_header ;
                GB_Operator op = GB_unop_identity (A->type, &op_header) ;
                ASSERT_OP_OK (op, "identity op for convert s2b", GB0) ;
                info = GB_convert_s2b_jit (Ax_new, Ab, op,
                    A, A_ek_slicing, A_ntasks, A_nthreads) ;
            }

            //------------------------------------------------------------------
            // via the generic kernel
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            { 
                // with user-defined types of other sizes
                #define GB_A_TYPE GB_void
                #undef  GB_COPY
                #define GB_COPY(Axnew,pnew,Ax,p)                         \
                    memcpy (Axnew +(pnew)*asize, Ax +(p)*asize, asize)
                #include "GB_convert_s2b_template.c"
                info = GrB_SUCCESS ;
            }
        }
    }

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // free prior content of A and transplant the new content
    //--------------------------------------------------------------------------

    if (in_place)
    { 
        // if in-place, remove A->x from A so it is not freed
        A->x = NULL ;
        A->x_shallow = false ;
    }

    GB_phybix_free (A) ;
    A->iso = A_iso ;        // OK: convert_s2b, keep iso

    A->b = Ab ; A->b_size = Ab_size ; A->b_shallow = false ;
    Ab = NULL ;

    A->x = Ax_keep ; A->x_size = Ax_size ; A->x_shallow = Ax_shallow ;

    A->nvals = anz - nzombies ;
    ASSERT (A->nzombies == 0) ;

    A->plen = -1 ;
    A->nvec = avdim ;
    A->nvec_nonempty = (avlen == 0) ? 0 : avdim ;

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (A, "A converted from sparse to bitmap", GB0) ;
    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

