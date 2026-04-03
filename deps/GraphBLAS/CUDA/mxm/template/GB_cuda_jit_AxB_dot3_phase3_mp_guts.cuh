
    //----------------------------------------------------------------------
    // trim X and Y
    //----------------------------------------------------------------------

    //try to trim tail of X

    while ( (pX_end - pX_start > shared_vector_size) 
         && (Yi[pY_end-1] < Xi[pX_end - shared_vector_size -1 ])  )
    {
       pX_end -= shared_vector_size;
    }

    //try to trim tail of Y
    while ( (pY_end - pY_start > shared_vector_size) 
         && (Xi[pX_end-1] < Yi[pY_end - shared_vector_size -1 ]) )
    {
       pY_end -= shared_vector_size;
    }

    int64_t Xnz = pX_end - pX_start ;

    int shared_steps_X = (Xnz + shared_vector_size -1)/shared_vector_size;

    int64_t step_end = (shared_steps_X <= 1? Xnz : shared_vector_size);

    while( (shared_steps_X>0) && (Yi[pY_start] > Xi[pX_start+ step_end-1]) )
    {  // Fast forward to skip empty intersections
       pX_start += step_end;
       Xnz = pX_end - pX_start ;
       shared_steps_X -= 1;
       step_end = (shared_steps_X <= 1? Xnz : shared_vector_size);
    }

    for ( int64_t kk = tid; kk< step_end; kk+= blockDim.x)
    {
        Xi_s[kk] = Xi[ kk + pX_start];
    }   
    this_thread_block().sync();
     
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

    for ( int64_t kk =tid; kk< step_end; kk+= blockDim.x)
    {
        Yi_s[kk] = Yi[ kk + pY_start];
    }   
    this_thread_block().sync();

    //----------------------------------------------------------------------
    // compute cij
    //----------------------------------------------------------------------

    //we want more than one intersection per thread
    while ( (shared_steps_X > 0) && (shared_steps_Y > 0) )
    {
        int64_t Xwork = pX_end - pX_start;
        int64_t Ywork = pY_end - pY_start;
        if ( shared_steps_X > 1) Xwork = shared_vector_size;  
        if ( shared_steps_Y > 1) Ywork = shared_vector_size;  
        int64_t nxy = Xwork + Ywork;

        // ceil Divide by 32 = blockDim.x :
        int work_per_thread = (nxy + blockDim.x -1)/blockDim.x;
        int diag     = GB_IMIN( work_per_thread*tid, nxy);
        int diag_end = GB_IMIN( diag + work_per_thread, nxy);

        // Ywork takes place of Ynz:
        int x_min = GB_IMAX( (diag - Ywork) , 0);

        //Xwork takes place of Xnz:
        int x_max = GB_IMIN( diag, Xwork);

        while ( x_min < x_max)
        {
            //binary search for correct diag break
            int pivot = (x_min +x_max) >> 1;
            int64_t Xpiv =  Xi_s[pivot] ;
            int64_t Ypiv = Yi_s[diag -pivot -1] ;

            x_min = (pivot + 1)* (Xpiv < Ypiv)  + x_min * (1 - (Xpiv < Ypiv));
            x_max = pivot * (1 - (Xpiv < Ypiv)) + x_max * (Xpiv < Ypiv);

        }
        int xcoord = x_min;
        int ycoord = diag -x_min -1;

        /* 
        //predictor-corrector search independent on each thread
        int xcoord = GB_IMAX(diag-1, 0); //predicted to be uniform distribution
        while ( Xi_s[xcoord] < Yi_s[diag-xcoord-1] && (xcoord<x_max) ) xcoord++; 
        while ( Xi_s[xcoord] > Yi_s[diag-xcoord-1] && (xcoord>x_min) ) xcoord--;

        int ycoord = diag -xcoord -1;
        */

        int64_t Xtest = Xi_s[xcoord] ;
        int64_t Ytest = Yi_s[ycoord] ;
        if ( (diag > 0) && (diag < nxy ) && (ycoord >= 0 ) && (Xtest == Ytest)) 
        { 
            diag--; //adjust for intersection incrementing both pointers 
        }
        // two start points are known now
        int tx_start = xcoord; // +pX_start;
        int ty_start = diag -xcoord; // +pY_start; 

        x_min = GB_IMAX( (diag_end - Ywork), 0); //Ywork replace Ynz
        x_max = GB_IMIN( diag_end, Xwork);      //Xwork replace Xnz

        while ( x_min < x_max) 
        {
            int pivot = (x_min +x_max) >> 1;
            int64_t Xpiv = Xi_s[pivot] ;
            int64_t Ypiv = Yi_s[diag_end -pivot -1] ;

            x_min = (pivot + 1)* (Xpiv < Ypiv)   + x_min * (1 - (Xpiv < Ypiv));
            x_max = pivot * (1 - (Xpiv < Ypiv))  + x_max * (Xpiv < Ypiv);
        }

        xcoord = x_min;
        ycoord = diag_end -x_min -1;


/*	    
        //predictor-corrector search independent on each thread
        xcoord = diag_end-1; //predicted to be uniform distribution
        while ( Xi_s[xcoord] < Yi_s[diag_end-xcoord-1] && (xcoord<x_max)) xcoord++; 
        while ( Xi_s[xcoord] > Yi_s[diag_end-xcoord-1] && (xcoord>x_min)) xcoord--;

        ycoord = diag_end -xcoord -1;
*/	   

        // two end points are known now
        int tx_end = xcoord; // +pX_start; 
        int ty_end = diag_end - xcoord; // + pY_start; 

        //merge-path dot product
        int64_t pX = tx_start;       // pX
        int64_t pY = ty_start;       // pY

        while ( pX < tx_end && pY < ty_end ) 
        {
            int64_t Xind = Xi_s[pX] ;
            int64_t Yind = Yi_s[pY] ;
            #if GB_IS_PLUS_PAIR_REAL_SEMIRING && GB_Z_IGNORE_OVERFLOW
                cij += (Xind == Yind) ;
            #else
                if (Xind == Yind)
                {
                    // cij += aki * bkj
                    #if MP_FLIP
                    GB_DOT_MERGE (pY + pY_start, pX + pX_start) ;
                    #else
                    GB_DOT_MERGE (pX + pX_start, pY + pY_start) ;
                    #endif
                    // TODO check terminal condition, using tile.any
                }
            #endif
            pX += (Xind <= Yind) ;
            pY += (Xind >= Yind) ;
        }
        GB_CIJ_EXIST_POSTCHECK ;

        this_thread_block().sync();

        if  (  (shared_steps_X >= 1) 
        && (shared_steps_Y >= 1) 
        && ( Xi_s[Xwork-1] == Yi_s[Ywork-1]) )
        {

            pX_start += shared_vector_size;
            shared_steps_X -= 1;
            if (shared_steps_X == 0) break;
            pY_start += shared_vector_size;
            shared_steps_Y -= 1;
            if (shared_steps_Y == 0) break;

            step_end = ( (pX_end - pX_start) < shared_vector_size ? (pX_end - pX_start) : shared_vector_size);
            while( (shared_steps_X>0) && (Yi[pY_start] > Xi[pX_start + step_end-1]) )
            { //fast forward
               pX_start += step_end;
               shared_steps_X -= 1;
               step_end = ( (pX_end - pX_start) < shared_vector_size ? (pX_end - pX_start) : shared_vector_size);
            }
            if (shared_steps_X == 0) break;

            for ( int64_t kk = tid; kk< step_end; kk+= blockDim.x)
            {
                Xi_s[kk] = Xi[ kk + pX_start];
            }   
            this_thread_block().sync();

            step_end = ( (pY_end - pY_start) < shared_vector_size ? (pY_end - pY_start) : shared_vector_size);
            while( (shared_steps_Y>0) && (Xi[pX_start] > Yi[pY_start + step_end-1]) )
            { //fast forward
               pY_start += step_end;
               shared_steps_Y -= 1;
               step_end = ( (pY_end - pY_start) < shared_vector_size ? (pY_end - pY_start) : shared_vector_size);
            }
            if (shared_steps_Y == 0) break;

            for ( int64_t kk = tid; kk< step_end; kk+= blockDim.x)
            {
                Yi_s[kk] = Yi[ kk + pY_start];
            }   
            this_thread_block().sync();

        } 
        else if ( (shared_steps_X >= 1) && (Xi_s[Xwork-1] < Yi_s[Ywork-1] ))
        {
            pX_start += shared_vector_size;
            shared_steps_X -= 1;
            if (shared_steps_X == 0) break;

            step_end= ( (pX_end - pX_start) < shared_vector_size ? (pX_end - pX_start) : shared_vector_size);
            while( (shared_steps_X>0) && (Yi[pY_start] > Xi[pX_start + step_end-1]) )
            { //fast forward
               pX_start += step_end;
               shared_steps_X -= 1;
               step_end= ( (pX_end - pX_start) < shared_vector_size ? (pX_end - pX_start) : shared_vector_size);
            }
            if (shared_steps_X == 0) break;

            for ( int64_t kk = tid; kk< step_end; kk+= blockDim.x)
            {
                Xi_s[kk] = Xi[ kk + pX_start];
            }   
            this_thread_block().sync();

        }
        else if ( (shared_steps_Y >= 1) && (Yi_s[Ywork-1] < Xi_s[Xwork-1]) )
        {
            pY_start += shared_vector_size;
            shared_steps_Y -= 1;
            if (shared_steps_Y == 0) break;

            step_end = ( (pY_end - pY_start) < shared_vector_size ? (pY_end - pY_start) : shared_vector_size);
            while( (shared_steps_Y>0) && (Xi[pX_start] > Yi[pY_start + step_end-1]) )
            { //fast forward
               pY_start += step_end;
               shared_steps_Y -= 1;
               step_end = ( (pY_end - pY_start) < shared_vector_size ? (pY_end - pY_start) : shared_vector_size);
            }
            if (shared_steps_Y == 0) break;

            for ( int64_t kk = tid; kk< step_end; kk+= blockDim.x)
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

