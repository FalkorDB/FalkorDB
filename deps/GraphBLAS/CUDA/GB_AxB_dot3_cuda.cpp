//------------------------------------------------------------------------------
// GB_AxB_dot3_cuda: compute C<M> = A'*B in parallel, on the GPU(s)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This function computes C<M>=A'*B on the GPUs.  The mask must be present,
// sparse or hypersparse, and not complemented.  The mask is always applied.  A
// and B can have any sparsity format.  C is computed as sparse or hypersparse,
// with the same format as M.

#include "GB_cuda.h"
extern "C"
{
    #include "GB_mxm.h"
}

#include "GB_jit_cache.h"
#include "GB_cuda_common_jitFactory.hpp"
#include "GB_cuda_reduce_jitFactory.hpp"
#include "GB_cuda_mxm_dot3_jitFactory.hpp"
#include "GB_cuda_type_wrap.hpp"
#include "test/GpuTimer.h"

/*
template<typename T, typename I>
void print_array(void *arr, I size, const char *name) {
    std::cout << "Printing " << name << std::endl;
    for(I i = 0; i < size; ++i) {
        std::cout << static_cast<T*>(arr)[i] << ", ";
    }
    std::cout << std::endl << "Done." << std::endl;
}
*/

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                               \
{                                                                       \
    /* FIXME: use a stream pool instead */                              \
    CU_OK (cudaStreamSynchronize(stream));                              \
    CU_OK (cudaStreamDestroy(stream));                                  \
    GB_FREE_WORK (&Nanobuckets, Nb_size) ;                              \
    GB_FREE_WORK (&Blockbucket, Bb_size) ;                              \
    GB_FREE_WORK (&Bucketp, Bup_size) ;                                 \
    GB_FREE_WORK (&offset, O_size) ;                                    \
    GB_FREE_WORK (&Bucket, Bu_size) ;                                   \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                                     \
{                                                                       \
    GB_FREE_WORKSPACE ;                                                 \
    GB_phybix_free (C) ;                                                \
}

//------------------------------------------------------------------------------
// GB_AxB_dot3_cuda
//------------------------------------------------------------------------------

GrB_Info GB_AxB_dot3_cuda           // C<M> = A'*B using dot product method
(
    GrB_Matrix C,                   // output matrix
    const GrB_Matrix M,             // mask matrix
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy               // if true, do z=fmult(b,a) vs fmult(a,b)
)
{

    // FIXME: pass in a stream instead, or checkout a stream
    cudaStream_t stream = NULL ;
    CU_OK (cudaStreamCreate(&stream));

    GpuTimer kernel_timer; 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // when CUDA is enabled, no static headers are used in all of GraphBLAS
    GrB_Info info ;
    ASSERT (C != NULL && !(C->static_header)) ;
    ASSERT (M != NULL && !(M->static_header)) ;
    ASSERT (A != NULL && !(A->static_header)) ;
    ASSERT (B != NULL && !(B->static_header)) ;

    ASSERT_MATRIX_OK (M, "M for dot3 cuda A'*B", GB0) ;
    ASSERT_MATRIX_OK (A, "A for dot3 cuda A'*B", GB0) ;
    ASSERT_MATRIX_OK (B, "B for dot3 cuda A'*B", GB0) ;

    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;

    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT (!GB_PENDING (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for dot3 numeric A'*B", GB0) ;

    ASSERT (A->vlen == B->vlen) ;
    GBURBLE ("(GPU dot3) ") ;
    //printf ("\nM -------------\n") ; GxB_Matrix_fprint (M, "M", GxB_SHORT, stdout) ;
    //printf ("\nA -------------\n") ; GxB_Matrix_fprint (A, "A", GxB_SHORT, stdout) ;
    //printf ("\nB -------------\n") ; GxB_Matrix_fprint (B, "B", GxB_SHORT, stdout) ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int64_t *Nanobuckets = NULL ; size_t Nb_size  = 0 ;
    int64_t *Blockbucket = NULL ; size_t Bb_size  = 0 ;
    int64_t *Bucket = NULL      ; size_t Bu_size  = 0 ;
    int64_t *Bucketp = NULL     ; size_t Bup_size = 0 ;
    int64_t *offset = NULL      ; size_t O_size   = 0 ;

    int device = -1;

    // FIXME: control the GPU to use via the descriptor
    CU_OK (cudaSetDevice( 0 ));
    CU_OK (cudaGetDevice(&device));

    //--------------------------------------------------------------------------
    // get M
    //--------------------------------------------------------------------------

    const int64_t mvlen = M->vlen ;
    const int64_t mvdim = M->vdim ;
    const int64_t mnz = GB_nnz (M) ;
    const int64_t mnvec = M->nvec ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE( M ) ;

    const int64_t anz = GB_nnz (A) ;
    const int64_t anvec = A->nvec ;

    const int64_t bnz = GB_nnz (B) ;
    const int64_t bnvec = B->nvec ;

    //--------------------------------------------------------------------------
    // allocate C, the same size and # of entries as M
    //--------------------------------------------------------------------------

    // FUTURE: ctype need not be the op->ztype
    GrB_Type ctype = semiring->add->op->ztype ;
    int64_t cvlen = mvlen ;
    int64_t cvdim = mvdim ;
    int64_t cnz = mnz ;
    int64_t cnvec = mnvec ;

    int M_sparsity = (M_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE ;
    int C_sparsity = M_sparsity ;
    bool C_iso = false ;    // FIXME: pass in C_iso and cscalar
    bool C_in_iso = false ;    // FIXME: pass in C_in_iso and cscalar
    info = GB_new_bix (&C, // sparse or hyper (from M), existing header
        ctype, cvlen, cvdim, GB_Ap_malloc, true,
        M_sparsity, false, M->hyper_switch, cnvec,
        cnz+1,  // add one to cnz for GB_cumsum of Cwork 
        true, C_iso) ;

    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (info) ;
    }

//  try this with GB_Ap_null, above in GB_new_bix
//  C->p = M->p ; C->p_shallow = true ;
//  C->h = M->h ; C->h_shallow = true ;

    //--------------------------------------------------------------------------
    // Pre-fetch arrays that will be used on the device
    //--------------------------------------------------------------------------

    // GB_cuda_matrix_advise (C, cnvec, cnz, which, what, device)
    // advise C
    CU_OK (cudaMemAdvise (C->p, (cnvec+1) * sizeof ( int64_t),
        cudaMemAdviseSetPreferredLocation, device)) ;
    if (M_is_hyper)
    { 
        CU_OK (cudaMemAdvise (C->h, cnvec * sizeof ( int64_t),
            cudaMemAdviseSetPreferredLocation, device)) ;
    }
    CU_OK (cudaMemAdvise (C->i, (cnz+1) * sizeof ( int64_t),
        cudaMemAdviseSetPreferredLocation, device)) ;
    if (!C_iso)
    {
        CU_OK (cudaMemAdvise (C->x, (cnz+1) * C->type->size ,
            cudaMemAdviseSetPreferredLocation, device)) ;
    }

    // prefetch M (if M hypersparse: using M->h not M->Y)
    GB_OK (GB_cuda_matrix_prefetch (M,
        Mask_struct ? GB_PREFETCH_PHBI : GB_PREFETCH_PHBIX, device, stream)) ;

    //--------------------------------------------------------------------------
    // copy Mp and Mh into C
    //--------------------------------------------------------------------------

    // FIXME: use shallow?
    CU_OK (cudaMemcpyAsync (C->p, M->p, (cnvec+1) * sizeof (int64_t),
        cudaMemcpyDefault, stream)) ;
    if (M_is_hyper)
    { 
        CU_OK (cudaMemcpyAsync (C->h, M->h, cnvec * sizeof (int64_t),
            cudaMemcpyDefault, stream)) ;
    }

    C->nvals = cnz ;
    C->magic = GB_MAGIC ;
    C->nvec_nonempty = M->nvec_nonempty ;
    C->jumbled = GB_JUMBLED (M) ;   // C is jumbled if M is jumbled

    GBURBLE ("(GPU C created and copied from M) ") ;

    //--------------------------------------------------------------------------
    // stringify the semiring and the mask
    //--------------------------------------------------------------------------

    GB_cuda_mxm_factory my_mxm_spec = GB_cuda_mxm_factory ( ) ;

    // (1) create the mxm code and name
    my_mxm_spec.mxm_factory ( C_iso, C_in_iso, C_sparsity, ctype,
        M, Mask_struct, false, semiring, flipxy, A, B) ;

    // (2) ensure the jitifier has "GB_mxm_[my_mxm_spec.sr_code].h"
    jit::GBJitCache filecache = jit::GBJitCache::Instance() ;
    filecache.getFile (my_mxm_spec) ;

    GBURBLE ("(GPU stringified srcode = %lu)\n", my_mxm_spec.sr_code) ;

    //--------------------------------------------------------------------------
    // get A and B
    //--------------------------------------------------------------------------

    // FIXME: add acode, bcode to the GB_cuda_mxm_factory object
    int acode = GB_RSHIFT (my_mxm_spec.sr_code, 12, 4) ;   // if 0: A is pattern
    int bcode = GB_RSHIFT (my_mxm_spec.sr_code,  8, 4) ;   // if 0: B is pattern

    bool A_is_sparse = GB_IS_SPARSE (A) ;
    bool A_is_hyper  = GB_IS_HYPERSPARSE (A) ;
    bool A_is_bitmap = GB_IS_BITMAP (A) ;
    bool A_is_full   = GB_IS_FULL (A) ;
    bool A_is_sparse_or_hyper = A_is_sparse || A_is_hyper ;
    bool A_is_bitmap_or_full  = A_is_bitmap || A_is_full  ;
    bool A_is_pattern = (acode == 0) ;

    bool B_is_sparse = GB_IS_SPARSE (B) ;
    bool B_is_hyper  = GB_IS_HYPERSPARSE (B) ;
    bool B_is_bitmap = GB_IS_BITMAP (B) ;
    bool B_is_full   = GB_IS_FULL (B) ;
    bool B_is_sparse_or_hyper = B_is_sparse || B_is_hyper ;
    bool B_is_bitmap_or_full  = B_is_bitmap || B_is_full  ;
    bool B_is_pattern = (bcode == 0) ;

    // M might be very very sparse.  A(:,i) is not needed if M(:,i) is empty.
    // Likewise, B(:,j) is not needed if M(:,j) is empty.  For now, try this
    // heuristic:  if M is hypersparse, then do not prefetch A->b or A->x.  

    int prefetch_b = (M_is_hyper) ? 0 : GB_PREFETCH_B ;
    int prefetch_x = (M_is_hyper) ? 0 : GB_PREFETCH_X ;
    int prefetch_pybi = GB_PREFETCH_PYI + prefetch_b ;

    // prefetch A (if A hypersparse: using A->Y)
    GB_OK (GB_cuda_matrix_prefetch (A, prefetch_pybi +
        (A_is_pattern ? 0 : prefetch_x), device, stream)) ;

    // prefetch B (if B hypersparse: using B->Y)
    GB_OK (GB_cuda_matrix_prefetch (B, prefetch_pybi +
        (B_is_pattern ? 0 : prefetch_x), device, stream)) ;

    //--------------------------------------------------------------------------
    // C<M>=A'*B via jitified kernels
    //--------------------------------------------------------------------------

    if (A_is_bitmap_or_full && B_is_bitmap_or_full)
    {

        //----------------------------------------------------------------------
        // (full or bitmap) times (full or bitmap)
        //----------------------------------------------------------------------

        dense_phase1launchFactory dp1lf(my_mxm_spec);

        GBURBLE ("(GPU phase1 start nblk = %d) ",
            dp1lf.get_number_of_blocks(M)) ;
        kernel_timer.Start();
            dp1lf.jitGridBlockLaunch(C, M, A, B, stream);
            CU_OK (cudaStreamSynchronize(stream));
        kernel_timer.Stop();
        GBURBLE ("(GPU phase1 done %12.6g ms )\n", kernel_timer.Elapsed()) ;

        mxm_dense_launchFactory mdlf(my_mxm_spec);
        GBURBLE ("(GPU Dense full x full launch ) ") ;
        kernel_timer.Start();
            mdlf.jitGridBlockLaunch( C, M, A, B, stream);
            CU_OK (cudaStreamSynchronize(stream));  // only for timing
        kernel_timer.Stop();
        GBURBLE ("(GPU Dense full x full done %12.6g ms, rate=%12.6g)\n", 
               kernel_timer.Elapsed(), (mnvec)/(1000*kernel_timer.Elapsed())) ;  

    }
    else
    {

        //----------------------------------------------------------------------
        // (sparse or hyper) times (sparse or hyper)
        // (sparse or hyper) times (bitmap or full)
        // (bitmap or full) times (sparse or hyper)
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // construct the tasks for phase1 and phase2
        //----------------------------------------------------------------------

        // on the CPU: nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;
        // on the GPU:
        phase1launchFactory p1lf(my_mxm_spec);
        phase2launchFactory p2lf;
        phase2endlaunchFactory p2elf;

        // # of threads in phase1 and phase2 kernel launches are related
        // # by the size of the warp.  ph2_task = ph1_task/32 for example
        int nthrd = p2lf.get_threads_per_block();
        int ntasks = p2elf.get_number_of_blocks(M);

        int64_t nanobuckets_size = NBUCKETS * nthrd * ntasks;
        int64_t blockbuckets_size = NBUCKETS * ntasks;

        Nanobuckets = GB_MALLOC_WORK (nanobuckets_size, int64_t, &Nb_size) ;
        Blockbucket = GB_MALLOC_WORK (blockbuckets_size, int64_t, &Bb_size) ;
        Bucketp = GB_MALLOC_WORK (NBUCKETS+1, int64_t, &Bup_size) ;
        offset = GB_MALLOC_WORK (NBUCKETS, int64_t, &O_size) ;
        Bucket = GB_MALLOC_WORK (mnz, int64_t, &Bu_size) ;

        if (Nanobuckets == NULL || Blockbucket == NULL || Bucketp == NULL
            || Bucket == NULL || offset == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        // FIXME: do async with streams
        // FIXME: do we need any of these?
        //CU_OK (cudaMemsetAsync(Nanobuckets, 0,
        //    nanobuckets_size * sizeof(int64_t), stream));
        //CU_OK (cudaMemsetAsync(Blockbucket, 0,
        //    blockbuckets_size * sizeof(int64_t), stream));
        CU_OK (cudaMemsetAsync(Bucketp, 0,
            (NBUCKETS+1) * sizeof(int64_t), stream));
        CU_OK (cudaMemsetAsync(offset, 0,
            NBUCKETS * sizeof(int64_t), stream));
        //CU_OK (cudaMemsetAsync(Bucket, 0,
        //    mnz * sizeof(int64_t), stream));

        //----------------------------------------------------------------------
        // phase1 and phase2: place each C(i,j) in a bucket
        //----------------------------------------------------------------------

        CU_OK (cudaMemAdvise( Bucketp, (NBUCKETS+1) * sizeof ( int64_t),
            cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CU_OK (cudaMemAdvise( Bucketp, (NBUCKETS+1) * sizeof ( int64_t),
            cudaMemAdviseSetAccessedBy, device));

        CU_OK (cudaMemAdvise( offset, NBUCKETS * sizeof ( int64_t),
            cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CU_OK (cudaMemAdvise( offset, NBUCKETS * sizeof ( int64_t),
            cudaMemAdviseSetAccessedBy, device));

        //----------------------------------------------------------------------
        // phase1: assign each C(i,j) to a bucket, and count them
        //----------------------------------------------------------------------

        GBURBLE ("(GPU phase1 start nblk = %d) ", p1lf.get_number_of_blocks(M));
        kernel_timer.Start();
        p1lf.jitGridBlockLaunch(Nanobuckets, Blockbucket, C, M, A, B, stream);
        CU_OK (cudaStreamSynchronize(stream));
        kernel_timer.Stop();

        GBURBLE ("(GPU phase1 done %12.6g ms )\n", kernel_timer.Elapsed()) ;

        //----------------------------------------------------------------------
        // phase2: cumsum across the blockbuckets, propagate to thread level
        //----------------------------------------------------------------------

        GBURBLE ("(GPU phase2 start nblk=%d ) ", ntasks) ;

        kernel_timer.Start();
        p2lf.jitGridBlockLaunch(Blockbucket, offset, M, stream);
        kernel_timer.Stop();

        CU_OK (cudaStreamSynchronize(stream));

        int64_t s= offset[0];
        C->nzombies = s;
        bool all_in_one = false;
        for ( int bucket = 1 ; bucket < NBUCKETS+1; ++bucket)
        {
            Bucketp[bucket] = s; 
            s += offset[bucket];
            if ( (Bucketp[bucket] - Bucketp[bucket-1] ) == mnz )
            {
                all_in_one = true;
            }
        }

        GBURBLE ("(GPU phase2 done %12.6g ms )\n", kernel_timer.Elapsed()) ;

        if (!all_in_one) 
        {
            GBURBLE ("(GPU phase2end start nblk=%d) ",  ntasks) ;

            kernel_timer.Start();
            p2elf.jitGridBlockLaunch(Nanobuckets, Blockbucket,
                             Bucketp, Bucket, offset, C, M, stream);

            CU_OK (cudaStreamSynchronize(stream));
            kernel_timer.Stop();
            GBURBLE ("(GPU phase2end done %12.6g ms)\n",kernel_timer.Elapsed());
        }

        //----------------------------------------------------------------------
        // phase3: do the numerical work
        //----------------------------------------------------------------------

        for ( int bucket = 1 ; bucket < NBUCKETS; ++bucket)
        {
            int64_t start = Bucketp[bucket];
            int64_t end   = Bucketp[bucket + 1 ];
            if (end - start > 0)
            {
                // TODO: Use stream pool
                phase3launchFactory p3lf(my_mxm_spec, (GB_bucket_code)bucket);
                GBURBLE ("(GPU phase3 bucket %d launch ) ", bucket) ;
                kernel_timer.Start();
                p3lf.jitGridBlockLaunch(start, end, Bucketp, Bucket,
                    C, M, A, B, stream);
                CU_OK (cudaStreamSynchronize(stream));  // only for timing
                kernel_timer.Stop();
                GBURBLE ("(GPU phase3 bucket %d done %12.6g ms, rate=%12.6g)\n",
                    bucket, kernel_timer.Elapsed(),
                    (end-start)/(1000*kernel_timer.Elapsed())) ; 
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    return GrB_SUCCESS; 
}

