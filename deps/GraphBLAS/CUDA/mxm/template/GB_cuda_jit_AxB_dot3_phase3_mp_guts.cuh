//------------------------------------------------------------------------------
// CUDA/mxm/template/GB_cuda_jit_AxB_dot3_phase3_mp_guts.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Merge-path-based dot product, where a single threadblock computes the
// complete dot product of one entry in C, cij = sum (A (:,i)*B(:,j)).

// See https://dl.acm.org/doi/10.1145/2304576.2304621 for a description of the
// merge-path method.  The method is modified here because the dot product is a
// set intersection, not a set union.

// The two input lists Ai [pA_start:pA_end-1] and Bi [pB_start:pB_end-1] are
// sorted in monotonically increasing order, with no duplicates.  The numerical
// values are in the same positions in Ax and Bx, but this is handled by the
// GB_DOT_MERGE macro below (which also handles the multiply/add operators).

//------------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // trim X and Y
    //--------------------------------------------------------------------------

    // Whole chunks of the tail ends of both Ai and Bi can be trimmed, if the
    // first entry in a chunk comes after the last entry in the other list.
    // This can only be done because a set intersection is being computed.

    // try to trim tail of X
    while ( (pX_end - pX_start > shared_vector_size)
         && (Yi[pY_end-1] < Xi[pX_end - shared_vector_size -1 ])  )
    {
       pX_end -= shared_vector_size ;
    }

    // try to trim tail of Y
    while ( (pY_end - pY_start > shared_vector_size)
         && (Xi[pX_end-1] < Yi[pY_end - shared_vector_size -1 ]) )
    {
       pY_end -= shared_vector_size ;
    }

    //--------------------------------------------------------------------------
    // load Xi_s with the first chunk of entries
    //--------------------------------------------------------------------------

    int64_t Xnz = pX_end - pX_start ;

    // fixme: rename shared_vector_size to CHUNK_SIZE to be consistent
    // with other CUDA kernels.
    // fixme: use " >> LOG2_SHARED_SIZE" instead of division
    int shared_steps_X = (Xnz + shared_vector_size -1)/shared_vector_size;

    int64_t step_end = (shared_steps_X <= 1? Xnz : shared_vector_size);

    while( (shared_steps_X>0) && (Yi[pY_start] > Xi[pX_start+ step_end-1]) )
    {  // Fast forward to skip empty intersections
       pX_start += step_end;
       Xnz = pX_end - pX_start ;
       shared_steps_X -= 1;
       step_end = (shared_steps_X <= 1? Xnz : shared_vector_size);
    }

    // Load the chunk: Xi_s [0:step_end-1] = Xi [pX_start + (0:step_end-1)]
    for (int64_t kk = threadIdx.x ; kk < step_end ; kk+= blockDim.x)
    {
        Xi_s [kk] = Xi [kk + pX_start] ;
    }
    this_thread_block().sync(); // FIXME is this needed? See next sync below.

    //--------------------------------------------------------------------------
    // load Yi_s with the first chunk of entries
    //--------------------------------------------------------------------------

    int64_t Ynz = pY_end - pY_start;          // Ynz

    int shared_steps_Y = (Ynz + shared_vector_size -1)/shared_vector_size;
    step_end = (shared_steps_Y <= 1 ? Ynz : shared_vector_size);

    while( (shared_steps_Y>0) && (Xi[pX_start] > Yi[pY_start + step_end-1]) )
    {  //Fast forward to skip
       pY_start+= step_end;
       Ynz = pY_end - pY_start;
       shared_steps_Y -= 1;
       step_end = (shared_steps_Y <= 1 ? Ynz : shared_vector_size);
    }

    // Load the chunk: Yi_s [0:step_end-1] = Yi [pY_start + (0:step_end-1)]
    for (int64_t kk = threadIdx.x ; kk < step_end ; kk += blockDim.x)
    {
        Yi_s [kk] = Yi [kk + pY_start] ;
    }
    this_thread_block().sync();

    //--------------------------------------------------------------------------
    // compute cij
    //--------------------------------------------------------------------------

    //we want more than one intersection per thread
    while ( (shared_steps_X > 0) && (shared_steps_Y > 0) )
    {

        // determine the # of entries in X and Y to handle in this pass
        // for this entire threadblock
        int64_t Xwork = pX_end - pX_start;
        int64_t Ywork = pY_end - pY_start;
        if ( shared_steps_X > 1) Xwork = shared_vector_size;
        if ( shared_steps_Y > 1) Ywork = shared_vector_size;
        int64_t nxy = Xwork + Ywork;

        // determine the thread-specific work to do for this pass
        // work_per_thread = ceil (nxy / blockDim.x) :
        int work_per_thread = (nxy + blockDim.x -1)/blockDim.x;
        int diag     = GB_IMIN( work_per_thread*threadIdx.x , nxy);
        int diag_end = GB_IMIN( diag + work_per_thread, nxy);

        //----------------------------------------------------------------------
        // each thread searches for its starting point
        //----------------------------------------------------------------------

        // Ywork takes place of Ynz:
        int x_min = GB_IMAX( (diag - Ywork) , 0);

        //Xwork takes place of Xnz:
        int x_max = GB_IMIN( diag, Xwork);

        while ( x_min < x_max)
        {
            //binary search for correct diag break
            int pivot = (x_min +x_max) >> 1;
            int64_t Xpiv = Xi_s [pivot] ;
            int64_t Ypiv = Yi_s [diag -pivot -1] ;

            // compute the following but without if/else:
            // if (Xpiv < Ypiv) x_min = pivot + 1 ; else x_max = pivot
            x_min = (pivot + 1)* (Xpiv < Ypiv)  + x_min * (1 - (Xpiv < Ypiv));
            x_max = pivot * (1 - (Xpiv < Ypiv)) + x_max * (Xpiv < Ypiv);

        }
        int xcoord = x_min;
        int ycoord = diag -x_min -1;

        // FIXME: this is necessary for correct operation of the mergepath
        // method, but this revision is NOT done for tx_end and ty_end. Why?
        int64_t Xtest = Xi_s [xcoord] ;
        int64_t Ytest = Yi_s [ycoord] ; // FIXME: out of bounds if ycoord < 0
        if ( (diag > 0) && (diag < nxy ) && (ycoord >= 0 ) && (Xtest == Ytest))
        {
            diag--; //adjust for intersection incrementing both pointers
        }

        // two start points are known now
        int tx_start = xcoord; // +pX_start;
        int ty_start = diag -xcoord; // +pY_start;

        //----------------------------------------------------------------------
        // each thread searches for its ending point
        //----------------------------------------------------------------------

        x_min = GB_IMAX( (diag_end - Ywork), 0); //Ywork replace Ynz
        x_max = GB_IMIN( diag_end, Xwork);      //Xwork replace Xnz

        while ( x_min < x_max)
        {
            int pivot = (x_min +x_max) >> 1;
            int64_t Xpiv = Xi_s [pivot] ;
            int64_t Ypiv = Yi_s [diag_end -pivot -1] ;

            // compute the following but without if/else:
            // if (Xpiv < Ypiv) x_min = pivot + 1 ; else x_max = pivot
            // fixme: use a helper static inline function to do this, or a macro
            // Question: Hard to read; why not use a ternary op instead?
            x_min = (pivot + 1)* (Xpiv < Ypiv)   + x_min * (1 - (Xpiv < Ypiv));
            x_max = pivot * (1 - (Xpiv < Ypiv))  + x_max * (Xpiv < Ypiv);
        }

        xcoord = x_min;
        ycoord = diag_end -x_min -1;

        // FIXME doesn't diag_end need revision just like (diag--) above?

        // two end points are known now
        int tx_end = xcoord; // +pX_start;
        // Question: why isn't ty_end the same as ycoord??
        int ty_end = diag_end - xcoord; // + pY_start;

        //----------------------------------------------------------------------
        // merge-path dot product for each thread
        //----------------------------------------------------------------------

        int64_t pX = tx_start;       // pX
        int64_t pY = ty_start;       // pY

        while ( pX < tx_end && pY < ty_end )
        {
            int64_t ia = Xi_s [pX] ;
            int64_t ib = Yi_s [pY] ;
            #if GB_IS_PLUS_PAIR_REAL_SEMIRING && GB_Z_IGNORE_OVERFLOW
                cij += (ia == ib) ;
            #else
                if (ia == ib)
                {
                    // cij += A(ia,i) * B(ia,j)
                    #if MP_FLIP
                    // fixme: no need for this case of MP_FLIP; see the caller
                    GB_DOT_MERGE (pY + pY_start, pX + pX_start, ia) ;
                    #else
                    GB_DOT_MERGE (pX + pX_start, pY + pY_start, ia) ;
                    #endif
                    // TODO check terminal condition, using tile.any
                }
            #endif
            pX += (ia <= ib) ;
            pY += (ia >= ib) ;
        }

        // FIXME: why is this done here and not at the very end?
        GB_CIJ_EXIST_POSTCHECK ;

        this_thread_block().sync();

        //----------------------------------------------------------------------
        // load the next chunk of X and Y into Xi_s and/or Yi_s
        //----------------------------------------------------------------------

        if  (  (shared_steps_X >= 1) && (shared_steps_Y >= 1)
            && ( Xi_s[Xwork-1] == Yi_s[Ywork-1]) )
        {

            //------------------------------------------------------------------
            // both chunks of X and Y are exhausted; reload both of them
            //------------------------------------------------------------------

            pX_start += shared_vector_size;
            shared_steps_X -= 1;
            if (shared_steps_X == 0) break;
            pY_start += shared_vector_size;
            shared_steps_Y -= 1;
            if (shared_steps_Y == 0) break;

            step_end = ( (pX_end - pX_start) < shared_vector_size ?
                (pX_end - pX_start) : shared_vector_size);

            while( (shared_steps_X>0) &&
                (Yi[pY_start] > Xi[pX_start + step_end-1]) )
            { //fast forward
               pX_start += step_end;
               shared_steps_X -= 1;
               step_end = ( (pX_end - pX_start) < shared_vector_size ?
                    (pX_end - pX_start) : shared_vector_size);
            }
            if (shared_steps_X == 0) break;

            // Xi_s [0:step_end-1] = Xi [pX_start + (0:step_end-1)]
            for (int64_t kk = threadIdx.x ; kk < step_end ; kk += blockDim.x)
            {
                Xi_s[kk] = Xi[ kk + pX_start];
            }
            this_thread_block().sync();

            step_end = ( (pY_end - pY_start) < shared_vector_size ?
                (pY_end - pY_start) : shared_vector_size);

            while( (shared_steps_Y>0) &&
                (Xi[pX_start] > Yi[pY_start + step_end-1]) )
            { //fast forward
               pY_start += step_end;
               shared_steps_Y -= 1;
               step_end = ( (pY_end - pY_start) < shared_vector_size ?
                    (pY_end - pY_start) : shared_vector_size);
            }
            if (shared_steps_Y == 0) break;

            // Yi_s [0:step_end-1] = Yi [pY_start + (0:step_end-1)]
            for (int64_t kk = threadIdx.x ; kk < step_end ; kk += blockDim.x)
            {
                Yi_s[kk] = Yi[ kk + pY_start];
            }
            this_thread_block().sync();

        }
        else if ( (shared_steps_X >= 1) && (Xi_s[Xwork-1] < Yi_s[Ywork-1] ))
        {

            //------------------------------------------------------------------
            // the X chunk is exhausted but the Y chunk is not; load more X
            //------------------------------------------------------------------

            // Question: isn't the Y chunk in Yi_s already partially consumed??
            // How is this accounted for in the next iteration, which uses
            // all of Yi_s [0..shared_vector_size-1] ??

            pX_start += shared_vector_size;
            shared_steps_X -= 1;
            if (shared_steps_X == 0) break;

            step_end= ( (pX_end - pX_start) < shared_vector_size ?
                (pX_end - pX_start) : shared_vector_size);

            while( (shared_steps_X>0) &&
                (Yi[pY_start] > Xi[pX_start + step_end-1]) )
            { //fast forward
               pX_start += step_end;
               shared_steps_X -= 1;
               step_end = ( (pX_end - pX_start) < shared_vector_size ?
                    (pX_end - pX_start) : shared_vector_size);
            }
            if (shared_steps_X == 0) break;

            // Xi_s [0:step_end-1] = Xi [pX_start + (0:step_end-1)]
            for ( int64_t kk = threadIdx.x ; kk< step_end; kk+= blockDim.x)
            {
                Xi_s[kk] = Xi[ kk + pX_start];
            }
            this_thread_block().sync();

        }
        else if ( (shared_steps_Y >= 1) && (Yi_s[Ywork-1] < Xi_s[Xwork-1]) )
        {

            //------------------------------------------------------------------
            // the Y chunk is exhausted but the X chunk is not; load more Y
            //------------------------------------------------------------------

            // Question: isn't the X chunk in Xi_s already partially consumed??
            // How is this accounted for in the next iteration, which uses
            // all of Xi_s [0..shared_vector_size-1] ??

            pY_start += shared_vector_size;
            shared_steps_Y -= 1;
            if (shared_steps_Y == 0) break;

            step_end = ( (pY_end - pY_start) < shared_vector_size ?
                (pY_end - pY_start) : shared_vector_size);

            while( (shared_steps_Y>0) &&
                (Xi[pX_start] > Yi[pY_start + step_end-1]) )
            { //fast forward
               pY_start += step_end;
               shared_steps_Y -= 1;
               step_end = ( (pY_end - pY_start) < shared_vector_size ?
                    (pY_end - pY_start) : shared_vector_size);
            }
            if (shared_steps_Y == 0) break;

            // Yi_s [0:step_end-1] = Yi [pY_start + (0:step_end-1)]
            for ( int64_t kk = threadIdx.x ; kk< step_end; kk+= blockDim.x)
            {
                Yi_s[kk] = Yi[ kk + pY_start];
            }
            this_thread_block().sync();
        }
    } // end while shared_steps_X > 0 && shared_steps_Y >0

    //tile.sync( ) ;

#undef MP_FLIP

#undef pX
#undef pX_start
#undef pX_end
#undef Xi

#undef pY
#undef pY_start
#undef pY_end
#undef Yi

