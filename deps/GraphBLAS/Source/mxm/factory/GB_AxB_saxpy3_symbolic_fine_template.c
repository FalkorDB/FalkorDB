//------------------------------------------------------------------------------
// GB_AxB_saxpy3_symbolic_fine_template: symbolic analysis for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// phase1 of saxpy3 symbolic, for fine hash tasks.  This method must be done
// in a single thread when using GCC on the Power or s390x architectures.
// Otherwise, it is done in parallel.

{

    //----------------------------------------------------------
    // phase1: fine hash task, C<M>=A*B or C<!M>=A*B
    //----------------------------------------------------------

    // If M_in_place is true, this is skipped.  The mask M is dense, and is
    // used in-place.

    // The least significant 2 bits of Hf [hash] is the flag f, and the upper
    // bits contain h, as (h,f).  After this phase1, if M(i,j)=1 then the hash
    // table contains ((i+1),1) in Hf [hash] at some location.

    // Later, the flag values of f = 2 and 3 are also used.
    // Only f=1 is set in this phase.

    // h == 0,   f == 0: unoccupied and unlocked
    // h == i+1, f == 1: occupied with M(i,j)=1

    uint64_t *restrict Hf = (uint64_t *restrict) SaxpyTasks [taskid].Hf ;
    uint64_t hash_bits = (hash_size-1) ;
    // scan this task's M(:,j)
    for (int64_t pM = mystart ; pM < myend ; pM++)
    {
        GB_GET_M_ij (pM) ;              // get M(i,j)
        if (!mij) continue ;            // skip if M(i,j)=0
        uint64_t i = GBi_M (Mi, pM, mvlen) ;
        uint64_t i_mine = ((i+1) << 2) + 1 ;  // ((i+1),1)
        for (GB_HASH (i))
        { 
            // swap this task's hash entry into the hash table;
            // does the following using an atomic capture:
            uint64_t hf ;

            #ifdef GCC_PPC_BUG
            {
                // NOTE: the 64-bit atomic capture fails with gcc 9.5.0 and
                // 14.2.0 on the IBM Power8NVL and s903x.  It works with
                // ibm-clang and clang 18.1.8 on the same Power8NVL system.
                // This works but it may have performance issues since this is
                // fixed by using this algorithm with a single thread that does
                // all fine hash tasks:
                { hf = Hf [hash] ; Hf [hash] = i_mine ; }
            }
            #else
            {
                // This fails when using gcc on the Power8 or s390x, but it
                // works fine on all other architectures.  It also works with
                // clang on Power8 and x390x.
                GB_ATOMIC_CAPTURE_UINT64 (hf, Hf [hash], i_mine) ;
            }
            #endif

            // Other methods tried:
            // This fails too, when using gcc on the Power8:
            // __atomic_exchange (&(Hf [hash]), &i_mine, &hf, __ATOMIC_SEQ_CST);

            // This also fails:
            // hf = __atomic_exchange_n (&(Hf [hash]), &i_mine,
            //    __ATOMIC_SEQ_CST) ;
            // __sync_synchronize ( ) ;

            // This doesn't work; it requires amo.h which reports an error on
            // the Power8NVL.  The Power8 doesn't implement the v3.0 ISA,
            // required by amo_ldat_swap.
            // hf = i_mine ;
            // _amo_ldat_swap (&hf, &(Hf [hash])) ;

            // This printf makes the GB_ATOMIC_CAPTURE work, if it is added!!
            // printf("   Hf [%ld] = %ld, i = %ld, i_mine = %ld\n", hash,
            //      Hf [hash], i, i_mine) ;

            // Because of this failure, gcc must be used with caution when
            // compiling GraphBLAS on the Power or s390 processor.  It cannot
            // use the GB_ATOMIC_CAPTURE_UINT64 defined in
            // Source/omp/include/GB_atomics.h.  If GraphBLAS is compiled with
            // gcc on the Power or s390x systems, GCC_PPC_BUG is #defined to
            // avoid this bug.

            if (hf == 0) break ;        // success
            // i_mine has been inserted, but a prior entry was
            // already there.  It needs to be replaced, so take
            // ownership of this displaced entry, and keep
            // looking until a new empty slot is found for it.
            i_mine = hf ;
        }
    }
}

