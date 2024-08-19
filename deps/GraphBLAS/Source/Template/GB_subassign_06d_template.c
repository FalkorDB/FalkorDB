//------------------------------------------------------------------------------
// GB_subassign_06d_template: C<A> = A
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#undef  GB_FREE_ALL
#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;   \
}

{

    //--------------------------------------------------------------------------
    // get the inputs
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    #define Mask_struct GB_MASK_STRUCT
    #define C_is_bitmap GB_C_IS_BITMAP
    #define A_is_bitmap GB_A_IS_BITMAP
    #define A_is_full   GB_A_IS_FULL
    #define A_iso       GB_A_ISO
    #define GB_AX_MASK(Ax,pA,asize) GB_MCAST (((GB_M_TYPE *) Ax), pA, asize)
    #else
    const bool C_is_bitmap = GB_IS_BITMAP (C) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_full = GB_IS_FULL (A) ;
    const bool A_iso = A->iso ;
    const size_t asize = A->type->size ;
    #endif

    ASSERT (C_is_bitmap || GB_IS_FULL (C)) ;

    //--------------------------------------------------------------------------
    // Parallel: slice A into equal-sized chunks
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    GB_A_NHELD (anz) ;
    int A_ntasks, A_nthreads ;
    double work = anz + A->nvec ;
    chunk = 32 * chunk ;        // method 06d needs a larger chunk
    if (A_is_bitmap || A_is_full)
    { 
        // no need to construct tasks
        A_nthreads = GB_nthreads (work, chunk, nthreads_max) ;
        A_ntasks = (A_nthreads == 1) ? 1 : (8 * A_nthreads) ;
    }
    else
    { 
        GB_SLICE_MATRIX_WORK (A, 8, work, anz) ;
    }

    //--------------------------------------------------------------------------
    // get C and A
    //--------------------------------------------------------------------------

    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const int8_t  *restrict Ab = A->b ;
    const int64_t avlen = A->vlen ;

    // since A is the mask, if A->iso is true, Mask_struct has been set true
    ASSERT (GB_IMPLIES (A_iso, Mask_struct)) ;

    int8_t *restrict Cb = C->b ;
    const int64_t cvlen = C->vlen ;

    #ifdef GB_ISO_ASSIGN
    // C is iso, and A is either iso or effectively iso (with a single entry
    // and not in bitmap form).  This case is only used by GB_subassign_06d
    // directly, and it is not needed for any kernel (generic, factor, or JIT).
    ASSERT (C->iso) ;
    GB_A_NVALS (e) ;
    ASSERT (A_iso || (e == 1 && !A_is_bitmap)) ;
    ASSERT (Mask_struct) ;
    #else
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    GB_DECLAREC (cwork) ;
    if (A_iso)
    { 
        // get the iso value of A and typecast to C->type
        // cwork = (ctype) Ax [0]
        GB_COPY_aij_to_cwork (cwork, Ax, 0, true) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // C<A> = A
    //--------------------------------------------------------------------------

    int64_t cnvals = C->nvals ;     // for C bitmap

    // future:: divide this Templates into sub-Templates (Mask_struct, etc)
    if (Mask_struct)
    {

        //----------------------------------------------------------------------
        // C<A,struct> = A where A can be iso or non-iso; mask is structural
        //----------------------------------------------------------------------

        if (A_is_full)
        {

            //------------------------------------------------------------------
            // A is full: all entries present
            //------------------------------------------------------------------

            #ifndef GB_ISO_ASSIGN
            {
                int64_t p ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static)
                for (p = 0 ; p < anz ; p++)
                { 
                    // Cx [p] = Ax [p]
                    GB_COPY_aij_to_C (Cx, p, Ax, p, A_iso, cwork) ;
                }
            }
            #endif

            if (C_is_bitmap)
            { 
                GB_memset (Cb, 1, anz, A_nthreads) ;
                cnvals = anz ;
            }

        }
        else if (A_is_bitmap)
        {

            //------------------------------------------------------------------
            // A is bitmap
            //------------------------------------------------------------------

            if (C_is_bitmap)
            {

                //--------------------------------------------------------------
                // C is bitmap, A is bitmap
                //--------------------------------------------------------------

                int tid ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static) reduction(+:cnvals)
                for (tid = 0 ; tid < A_nthreads ; tid++)
                {
                    int64_t pA_start, pA_end, task_cnvals = 0 ;
                    GB_PARTITION (pA_start, pA_end, anz, tid, A_nthreads) ;
                    for (int64_t p = pA_start ; p < pA_end ; p++)
                    {
                        if (Ab [p])
                        { 
                            // Cx [p] = Ax [p]
                            #ifndef GB_ISO_ASSIGN
                            GB_COPY_aij_to_C (Cx, p, Ax, p, A_iso, cwork) ;
                            #endif
                            task_cnvals += (Cb [p] == 0) ;
                            Cb [p] = 1 ;
                        }
                    }
                    cnvals += task_cnvals ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C is hypersparse, sparse, or full, with all entries present
                //--------------------------------------------------------------

                #ifndef GB_ISO_ASSIGN
                {
                    // this method is used by LAGraph_bfs_parent when q is
                    // a bitmap and pi is full.
                    int64_t p ;
                    #pragma omp parallel for num_threads(A_nthreads) \
                        schedule(static)
                    for (p = 0 ; p < anz ; p++)
                    {
                        // Cx [p] = Ax [p]
                        if (Ab [p])
                        { 
                            GB_COPY_aij_to_C (Cx, p, Ax, p, A_iso, cwork) ;
                        }
                    }
                }
                #endif
            }

        }
        else
        {

            //------------------------------------------------------------------
            // A is hypersparse or sparse; C is full or a bitmap
            //------------------------------------------------------------------

            const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
            const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
            const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2;
            int taskid ;

            if (C_is_bitmap)
            {

                //--------------------------------------------------------------
                // C is bitmap, mask is structural
                //--------------------------------------------------------------

                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(dynamic,1) reduction(+:cnvals)
                for (taskid = 0 ; taskid < A_ntasks ; taskid++)
                {
                    // if kfirst > klast then taskid does no work at all
                    int64_t kfirst = kfirst_Aslice [taskid] ;
                    int64_t klast  = klast_Aslice  [taskid] ;
                    int64_t task_cnvals = 0 ;
                    // C<A(:,kfirst:klast)> = A(:,kfirst:klast)
                    for (int64_t k = kfirst ; k <= klast ; k++)
                    {
                        // get A(:,j), the kth vector of A
                        int64_t j = GBH_A (Ah, k) ;
                        GB_GET_PA (pA_start, pA_end, taskid, k,
                            kfirst, klast, pstart_Aslice,
                            GBP_A (Ap, k, avlen), GBP_A (Ap, k+1, avlen)) ;
                        // pC is the start of C(:,j)
                        int64_t pC = j * cvlen ;
                        // C<A(:,j),struct>=A(:,j) with C bitmap, A sparse
                        GB_PRAGMA_SIMD_REDUCTION (+,task_cnvals)
                        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                        { 
                            int64_t p = pC + Ai [pA] ;
                            // Cx [p] = Ax [pA]
                            #ifndef GB_ISO_ASSIGN
                            GB_COPY_aij_to_C (Cx, p, Ax, pA, A_iso, cwork) ;
                            #endif
                            task_cnvals += (Cb [p] == 0) ;
                            Cb [p] = 1 ;
                        }
                    }
                    cnvals += task_cnvals ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C is full, mask is structural
                //--------------------------------------------------------------

                #ifndef GB_ISO_ASSIGN
                {
                    #pragma omp parallel for num_threads(A_nthreads) \
                        schedule(dynamic,1)
                    for (taskid = 0 ; taskid < A_ntasks ; taskid++)
                    {
                        // if kfirst > klast then taskid does no work at all
                        int64_t kfirst = kfirst_Aslice [taskid] ;
                        int64_t klast  = klast_Aslice  [taskid] ;
                        // C<A(:,kfirst:klast)> = A(:,kfirst:klast)
                        for (int64_t k = kfirst ; k <= klast ; k++)
                        {
                            // get A(:,j), the kth vector of A
                            int64_t j = GBH_A (Ah, k) ;
                            GB_GET_PA (pA_start, pA_end, taskid, k,
                                kfirst, klast, pstart_Aslice,
                                GBP_A (Ap, k, avlen), GBP_A (Ap, k+1, avlen)) ;
                            // pC is the start of C(:,j)
                            int64_t pC = j * cvlen ;
                            // C<A(:,j),struct>=A(:,j) with C full, A sparse
                            GB_PRAGMA_SIMD_VECTORIZE
                            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                            { 
                                int64_t p = pC + Ai [pA] ;
                                // Cx [p] = Ax [pA]
                                GB_COPY_aij_to_C (Cx, p, Ax, pA, A_iso, cwork) ;
                            }
                        }
                    }
                }
                #endif
            }
        }

    }
    #ifndef GB_ISO_ASSIGN
    else
    {

        //----------------------------------------------------------------------
        // C<A> = A where A must be non-iso, and the mask is valued
        //----------------------------------------------------------------------

        if (A_is_full)
        {

            //------------------------------------------------------------------
            // A is full: all entries present
            //------------------------------------------------------------------

            if (C_is_bitmap)
            {

                //--------------------------------------------------------------
                // C is bitmap, A is full
                //--------------------------------------------------------------

                int tid ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static) reduction(+:cnvals)
                for (tid = 0 ; tid < A_nthreads ; tid++)
                {
                    int64_t pA_start, pA_end, task_cnvals = 0 ;
                    GB_PARTITION (pA_start, pA_end, anz, tid, A_nthreads) ;
                    for (int64_t p = pA_start ; p < pA_end ; p++)
                    {
                        if (GB_AX_MASK (Ax, p, asize))
                        { 
                            // Cx [p] = Ax [p]
                            GB_COPY_aij_to_C (Cx, p, Ax, p, false, cwork) ;
                            task_cnvals += (Cb [p] == 0) ;
                            Cb [p] = 1 ;
                        }
                    }
                    cnvals += task_cnvals ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C is hypersparse, sparse, or full, with all entries present
                //--------------------------------------------------------------

                int64_t p ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static)
                for (p = 0 ; p < anz ; p++)
                {
                    if (GB_AX_MASK (Ax, p, asize))
                    { 
                        // Cx [p] = Ax [p]
                        GB_COPY_aij_to_C (Cx, p, Ax, p, false, cwork) ;
                    }
                }
            }

        }
        else if (A_is_bitmap)
        {

            //------------------------------------------------------------------
            // A is bitmap
            //------------------------------------------------------------------

            if (C_is_bitmap)
            {

                //-------------------------------------------------------------
                // C is bitmap, A is bitmap
                //--------------------------------------------------------------

                int tid ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static) reduction(+:cnvals)
                for (tid = 0 ; tid < A_nthreads ; tid++)
                {
                    int64_t pA_start, pA_end, task_cnvals = 0 ;
                    GB_PARTITION (pA_start, pA_end, anz, tid, A_nthreads) ;
                    for (int64_t p = pA_start ; p < pA_end ; p++)
                    {
                        if (Ab [p] && GB_AX_MASK (Ax, p, asize))
                        { 
                            // Cx [p] = Ax [p]
                            GB_COPY_aij_to_C (Cx, p, Ax, p, false, cwork) ;
                            task_cnvals += (Cb [p] == 0) ;
                            Cb [p] = 1 ;
                        }
                    }
                    cnvals += task_cnvals ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C is hypersparse, sparse, or full, with all entries present
                //--------------------------------------------------------------

                int64_t p ;
                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(static)
                for (p = 0 ; p < anz ; p++)
                {
                    if (Ab [p] && GB_AX_MASK (Ax, p, asize))
                    { 
                        // Cx [p] = Ax [p]
                        GB_COPY_aij_to_C (Cx, p, Ax, p, false, cwork) ;
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // A is hypersparse or sparse; C is full or bitmap
            //------------------------------------------------------------------

            const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
            const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
            const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2;
            int taskid ;

            if (C_is_bitmap)
            {

                //--------------------------------------------------------------
                // C is bitmap
                //--------------------------------------------------------------

                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(dynamic,1) reduction(+:cnvals)
                for (taskid = 0 ; taskid < A_ntasks ; taskid++)
                {
                    // if kfirst > klast then taskid does no work at all
                    int64_t kfirst = kfirst_Aslice [taskid] ;
                    int64_t klast  = klast_Aslice  [taskid] ;
                    int64_t task_cnvals = 0 ;
                    // C<A(:,kfirst:klast)> = A(:,kfirst:klast)
                    for (int64_t k = kfirst ; k <= klast ; k++)
                    {
                        // get A(:,j), the kth vector of A
                        int64_t j = GBH_A (Ah, k) ;
                        GB_GET_PA (pA_start, pA_end, taskid, k,
                            kfirst, klast, pstart_Aslice,
                            GBP_A (Ap, k, avlen), GBP_A (Ap, k+1, avlen)) ;
                        // pC is the start of C(:,j)
                        int64_t pC = j * cvlen ;
                        // C<A(:,j),struct>=A(:,j) with C bitmap, A sparse
                        GB_PRAGMA_SIMD_REDUCTION (+,task_cnvals)
                        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                        {
                            if (GB_AX_MASK (Ax, pA, asize))
                            { 
                                int64_t p = pC + Ai [pA] ;
                                // Cx [p] = Ax [pA]
                                GB_COPY_aij_to_C (Cx, p, Ax, pA, A_iso, cwork) ;
                                task_cnvals += (Cb [p] == 0) ;
                                Cb [p] = 1 ;
                            }
                        }
                    }
                    cnvals += task_cnvals ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // C is full
                //--------------------------------------------------------------

                #pragma omp parallel for num_threads(A_nthreads) \
                    schedule(dynamic,1) reduction(+:cnvals)
                for (taskid = 0 ; taskid < A_ntasks ; taskid++)
                {
                    // if kfirst > klast then taskid does no work at all
                    int64_t kfirst = kfirst_Aslice [taskid] ;
                    int64_t klast  = klast_Aslice  [taskid] ;
                    // C<A(:,kfirst:klast)> = A(:,kfirst:klast)
                    for (int64_t k = kfirst ; k <= klast ; k++)
                    {
                        // get A(:,j), the kth vector of A
                        int64_t j = GBH_A (Ah, k) ;
                        GB_GET_PA (pA_start, pA_end, taskid, k,
                            kfirst, klast, pstart_Aslice,
                            GBP_A (Ap, k, avlen), GBP_A (Ap, k+1, avlen)) ;
                        // pC is the start of C(:,j)
                        int64_t pC = j * cvlen ;
                        // C<A(:,j),struct>=A(:,j) with C full, A sparse
                        GB_PRAGMA_SIMD_VECTORIZE
                        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
                        {
                            if (GB_AX_MASK (Ax, pA, asize))
                            { 
                                int64_t p = pC + Ai [pA] ;
                                // Cx [p] = Ax [pA]
                                GB_COPY_aij_to_C (Cx, p, Ax, pA, A_iso, cwork) ;
                            }
                        }
                    }
                }
            }
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // log the number of entries in the C bitmap
    //--------------------------------------------------------------------------

    if (C_is_bitmap)
    { 
        C->nvals = cnvals ;
    }

    GB_FREE_ALL ;
}

#undef GB_ISO_ASSIGN
#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

