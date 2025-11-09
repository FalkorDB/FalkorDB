//------------------------------------------------------------------------------
// GB_convert_b2s: construct triplets or CSC/CSR from bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Constructs a triplet or CSC/CSR form (in Cp, Ci, Cj, and Cx_new) from the
// bitmap input matrix A.  If A is iso or Cx_new is NULL then no values are
// extracted.  The iso case is handled by the caller.

// Ci, Cj, and Cx_new may be NULL.

// FUTURE: make a separate function for constructing triplets

#include "GB.h"
#include "jitifyer/GB_stringify.h"
#include "unaryop/GB_unop.h"
#define GB_FREE_ALL GB_FREE_MEMORY (&W, W_size) ;

GrB_Info GB_convert_b2s   // extract CSC/CSR or triplets from bitmap
(
    // outputs:
    void *Cp,                   // vector pointers for CSC/CSR form
    void *Ci,                   // indices for CSC/CSR or triplet form
    void *Cj,                   // vector indices for triplet form
    void *Cx_new,               // values for CSC/CSR or triplet form
    int64_t *cnvec_nonempty,    // # of non-empty vectors
    // inputs: not modified
    const bool Cp_is_32,        // if true, Cp is uint32_t; otherwise uint64_t
    const bool Ci_is_32,        // if true, Ci is uint32_t; otherwise uint64_t
    const bool Cj_is_32,        // if true, Cj is uint32_t; otherwise uint64_t
    const GrB_Type ctype,       // type of Cx
    const GrB_Matrix A,         // matrix to extract; not modified
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (Cp != NULL) ;           // must be provided on input, size avdim+1
    ASSERT_MATRIX_OK (A, "A for b2s", GB0) ;
    ASSERT_TYPE_OK (ctype, "ctype for b2s", GB0) ;
    ASSERT ((Cp_is_32 && A->nvals < UINT32_MAX) || !Cp_is_32) ;

    //--------------------------------------------------------------------------
    // get inputs and determine tasks
    //--------------------------------------------------------------------------

    void *W = NULL ; size_t W_size = 0 ;
    GB_IDECL (W , , u) ;
    GB_IDECL (Cp, , u) ; GB_IPTR (Cp, Cp_is_32) ;
    GB_IDECL (Ci, , u) ; GB_IPTR (Ci, Ci_is_32) ;
    GB_IDECL (Cj, , u) ; GB_IPTR (Cj, Cj_is_32) ;

    size_t psize = Cp_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    const int64_t avdim = A->vdim ;
    const int64_t avlen = A->vlen ;
    const size_t asize = A->type->size ;
    const int8_t *restrict Ab = A->b ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (avlen*avdim, chunk, nthreads_max) ;
    bool by_vector = (nthreads <= avdim) ;

    //--------------------------------------------------------------------------
    // count the entries in each vector
    //--------------------------------------------------------------------------

    if (by_vector)
    {

        //----------------------------------------------------------------------
        // compute all vectors in parallel (no workspace)
        //----------------------------------------------------------------------

        int64_t j ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (j = 0 ; j < avdim ; j++)
        {
            // ajnz = nnz (A (:,j))
            uint64_t ajnz = 0 ;
            int64_t pA_start = j * avlen ;
            for (int64_t i = 0 ; i < avlen ; i++)
            { 
                // see if A(i,j) is present in the bitmap
                int64_t p = i + pA_start ;
                ajnz += Ab [p] ;
                ASSERT (Ab [p] == 0 || Ab [p] == 1) ;
            }
            // Cp [j] = ajnz ;
            GB_ISET (Cp, j, ajnz) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // compute blocks of rows in parallel
        //----------------------------------------------------------------------

        // allocate one row of W per thread, each row of length avdim
        W = GB_MALLOC_MEMORY (nthreads * avdim, psize, &W_size) ;
        if (W == NULL)
        {
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_IPTR (W, Cp_is_32) ;

        //----------------------------------------------------------------------
        // count each block
        //----------------------------------------------------------------------

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            uint32_t *restrict Wtask32 = Cp_is_32 ? (W32 + tid * avdim) : NULL ;
            uint64_t *restrict Wtask64 = Cp_is_32 ? NULL : (W64 + tid * avdim) ;
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, avlen, tid, nthreads) ;
            for (int64_t j = 0 ; j < avdim ; j++)
            {
                // ajnz = nnz (A (istart:iend-1,j))
                uint64_t ajnz = 0 ;
                int64_t pA_start = j * avlen ;
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    // see if A(i,j) is present in the bitmap
                    int64_t p = i + pA_start ;
                    ajnz += Ab [p] ;
                    ASSERT (Ab [p] == 0 || Ab [p] == 1) ;
                }
                // Wtask [j] = ajnz ;
                GB_ISET (Wtask, j, ajnz) ;
            }
        }

        //----------------------------------------------------------------------
        // cumulative sum to compute nnz(A(:,j)) for each vector j
        //----------------------------------------------------------------------

        int64_t j ;
        if (Cp_is_32)
        {
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (j = 0 ; j < avdim ; j++)
            {
                uint32_t ajnz = 0 ;
                for (int tid = 0 ; tid < nthreads ; tid++)
                { 
                    uint32_t *restrict Wtask32 = W32 + tid * avdim ;
                    uint32_t c = Wtask32 [j] ;
                    Wtask32 [j] = ajnz ;
                    ajnz += c ;
                }
                Cp32 [j] = ajnz ;
            }
        }
        else
        {
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (j = 0 ; j < avdim ; j++)
            {
                uint64_t ajnz = 0 ;
                for (int tid = 0 ; tid < nthreads ; tid++)
                { 
                    uint64_t *restrict Wtask64 = W64 + tid * avdim ;
                    uint64_t c = Wtask64 [j] ;
                    Wtask64 [j] = ajnz ;
                    ajnz += c ;
                }
                Cp64 [j] = ajnz ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // cumulative sum of Cp 
    //--------------------------------------------------------------------------

    // This cannot overflow if Cp is uint32_t, because in that case A->nvals
    // is < UINT32_MAX (see assertion above).

    int nth = GB_nthreads (avdim, chunk, nthreads_max) ;
    GB_cumsum (Cp, Cp_is_32, avdim, cnvec_nonempty, nth, Werk) ;
    ASSERT (GB_IGET (Cp, avdim) == A->nvals) ;

    //--------------------------------------------------------------------------
    // gather the pattern and values from the bitmap
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;

    if (Cx_new == NULL || A->x == NULL || A->iso)
    { 

        //----------------------------------------------------------------------
        // via the symbolic kernel
        //----------------------------------------------------------------------

        #undef  GB_COPY
        #define GB_COPY(Cx,pC,Ax,pA)
        #include "convert/template/GB_convert_b2s_template.c"
        info = GrB_SUCCESS ;

    }
    else
    {

        //----------------------------------------------------------------------
        // via an inline kernel for types of size 1, 2, 4, 8, or 16
        //----------------------------------------------------------------------

        if (ctype == A->type)
        {

            #undef  GB_COPY
            #define GB_COPY(Cx,pC,Ax,pA) Cx [pC] = Ax [pA] ;

            #ifndef GBCOMPACT
            GB_IF_FACTORY_KERNELS_ENABLED
            { 
                switch (asize)
                {

                    case GB_1BYTE : // uint8, int8, bool, or 1-byte user
                        #define GB_C_TYPE uint8_t
                        #define GB_A_TYPE uint8_t
                        #include "convert/template/GB_convert_b2s_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    case GB_2BYTE : // uint16, int16, or 2-byte user-defined
                        #define GB_C_TYPE uint16_t
                        #define GB_A_TYPE uint16_t
                        #include "convert/template/GB_convert_b2s_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    case GB_4BYTE : // uint32, int32, float, or 4-byte user
                        #define GB_C_TYPE uint32_t
                        #define GB_A_TYPE uint32_t
                        #include "convert/template/GB_convert_b2s_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    case GB_8BYTE : // uint64, int64, double, float complex,
                             // or 8-byte user defined
                        #define GB_C_TYPE uint64_t
                        #define GB_A_TYPE uint64_t
                        #include "convert/template/GB_convert_b2s_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    case GB_16BYTE : // double complex or 16-byte user-defined
                        #define GB_C_TYPE GB_blob16
                        #define GB_A_TYPE GB_blob16
                        #include "convert/template/GB_convert_b2s_template.c"
                        info = GrB_SUCCESS ;
                        break ;

                    default:;
                }
            }
            #endif
        }

        //----------------------------------------------------------------------
        // via the JIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            struct GB_UnaryOp_opaque op_header ;
            GB_Operator op = GB_unop_identity (ctype, &op_header) ;
            info = GB_convert_b2s_jit (Cp, Ci, Cj, Cx_new,
                Cp_is_32, Ci_is_32, Cj_is_32, ctype, op, A, W, nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            GB_Type_code ccode = ctype->code ;
            GB_Type_code acode = A->type->code ;
            const size_t csize = ctype->size ;
            GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;
            #define GB_C_TYPE GB_void
            #define GB_A_TYPE GB_void
            #undef  GB_COPY
            #define GB_COPY(Cx,pC,Ax,pA) \
                cast_A_to_C (Cx +(pC)*csize, Ax +(pA)*asize, asize)
            #include "convert/template/GB_convert_b2s_template.c"
            info = GrB_SUCCESS ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (info) ;
}

