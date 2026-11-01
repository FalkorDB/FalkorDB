//------------------------------------------------------------------------------
// CUDA/builder/template/GB_cuda_construct_Cphix.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Constructs the output matrix C (Cp, Ch, Ci, Cx) from its coordinate form.

template
<
    typename T_Cp,          // type of Cp and JDeltaSum
    typename T_Cj,          // type of Cj
    typename T_Ci,          // type of Ci
    typename T_Cij,         // type of Cij (Key_out for builder)
    typename T_Cx,          // type of Cx
    typename T_Sx,          // type of Sx
    int chunksize,          // must match GB_cuda_construct_JDelta
    int log2_chunksize,     // log2 (chunksize)
    bool mtx_build,         // if true, construct Cp and Ch for a matrix
    bool construct_Ci,      // if true, construct Ci
    bool construct_Cx,      // if true, construct Cx
    // lambda function types; the caller need not declare these types
    typename T_F1,          // type of unload_Ci function
    typename T_F2,          // type of unload_Cj function
    typename T_F3           // type of unload_Cx function
>
__device__ void GB_cuda_construct_Cphix
(
    // outputs
    GrB_Matrix C,
    // inputs, not modified:
    uint16_t *JDelta,       // size nvals+1, in JDelta [-1..nvals-1]
    T_Cp *JDeltaSum,        // size nchunks+1
    T_Cij *Cij,             // size nvals+1: Cj [-1 ... nvals-1]
    T_Sx *Sx,               // size nvals+1: Sx [-1 ... nvals-1]
    int64_t nvals,          // # of entries in the matrix
    int64_t nchunks,
    T_F1 unload_Ci,         // lambda function to get i from Cij [p]
    T_F2 unload_Cj,         // lambda function to get j from Cij [p]
    T_F3 unload_Cx          // lambda function to get cij from Sx [p]
)
{

    //--------------------------------------------------------------------------
    // get C->p, C->h, and C->i. kT is 1-based but p is 0-based
    //--------------------------------------------------------------------------

    T_Cp *__restrict__ Cp = (T_Cp *) C->p ; Cp-- ;  // index with kT, as 1-based
    T_Cj *__restrict__ Ch = NULL ;
    if constexpr (mtx_build)
    {
        // Ch is not constructed if C is a vector
        Ch = (T_Cj *) C->h ; Ch-- ;                 // index with kT, as 1-based
    }
    T_Ci *__restrict__ Ci = NULL ;
    if constexpr (construct_Ci)
    {
        // eWiseAdd has already constructed C->i, so it skips this step
        Ci = (T_Ci *) C->i ;                        // index with p, as 0-based
    }
    T_Cx *__restrict__ Cx = NULL ;
    if constexpr (construct_Cx)
    {
        Cx = (T_Cx *) C->x ;                        // index with p, as 0-based
    }

    //--------------------------------------------------------------------------
    // copy the entries from (Cj) into Cp, Ch, and Ci
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks ;
                 chunk += gridDim.x)        // grid-stride loop
    {

        //----------------------------------------------------------------------
        // determine the properties of this chunk
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << log2_chunksize ;
        int64_t my_chunk_size ;
        int64_t plast = pfirst + chunksize ;
        plast = GB_IMIN (plast, nvals) ;
        my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // copy the entries, sum duplicates, and construct Cp and Ch
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)       // block-stride loop
        {

            int64_t p = pfirst + pdelta ;

            //------------------------------------------------------------------
            // copy the entries
            //------------------------------------------------------------------

            if constexpr (construct_Ci)
            {
                // eWiseAdd has already construct C->i so this can be skipped
                // for that case
                // Ci [p] = Cij [p].i ;
                Ci [p] = unload_Ci (Cij, p) ;
            }

            if constexpr (construct_Cx)
            {
                Cx [p] = unload_Cx (Sx, p) ;    // Tx [p] = (cast) Sx [p]
            }

            //------------------------------------------------------------------
            // construct Cp and Ch, if C is a matrix (skip if C is a vector)
            //------------------------------------------------------------------

            if constexpr (mtx_build)
            {
                T_Cp kT = JDelta [p  ] + JDeltaSum [chunk] ;
                T_Cp k0 = JDelta [p-1] + JDeltaSum [chunk - (pdelta == 0)] ;
                if (k0 < kT)
                {
                    // The p-th entry is leading entry of the kT-th vector of C
                    Cp [kT] = p ;       // p is already 0-based
                    // Ch [kT] = Cij [p].j ;
                    Ch [kT] = unload_Cj (Cij, p) ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize the last vector of C
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // C->nvec is 0-based, so increment Cp to undo the Cp-- done above
        Cp++ ;
        if constexpr (mtx_build)
        {
            // C is a matrix; log the end of its last vector
            Cp [C->nvec] = C->nvals ;
        }
        else
        {
            // C is a vector; assign all of Cp
            Cp [0] = 0 ;
            Cp [1] = C->nvals ;
        }
    }
}

