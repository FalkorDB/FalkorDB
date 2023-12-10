//------------------------------------------------------------------------------
// GB_AxB_saxpy3_symbolic: symbolic analysis for GB_AxB_saxpy3
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Symbolic analysis for C=A*B, C<M>=A*B or C<!M>=A*B, via GB_AxB_saxpy3.
// Coarse tasks compute nnz (C (:,j)) for each of their vectors j.  Fine tasks
// just scatter the mask M into the hash table.  This phase does not depend on
// the semiring, nor does it depend on the type of C, A, or B.  It does access
// the values of M, if the mask matrix M is present and not structural.

// If B is hypersparse, C must also be hypersparse.
// Otherwise, C must be sparse.

// If both A and B are bitmap/full for C=A*B or C<!M>=A*B, then saxpy3 is
// not used.  C is selected as bitmap instead.

#include "GB_AxB_saxpy3.h"

void GB_AxB_saxpy3_symbolic
(
    GrB_Matrix C,               // Cp is computed for coarse tasks
    const GrB_Matrix M,         // mask matrix M
    const bool Mask_comp,       // M complemented, or not
    const bool Mask_struct,     // M structural, or not
    const bool M_in_place,
    const GrB_Matrix A,         // A matrix; only the pattern is accessed
    const GrB_Matrix B,         // B matrix; only the pattern is accessed
    GB_saxpy3task_struct *SaxpyTasks,     // list of tasks, and workspace
    const int ntasks,           // total number of tasks
    const int nfine,            // number of fine tasks
    const int nthreads          // number of threads
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_ZOMBIES (M)) ; 
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_PENDING (M)) ; 

    ASSERT (!GB_ZOMBIES (A)) ; 
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ; 

    ASSERT (!GB_ZOMBIES (B)) ; 
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_PENDING (B)) ; 

    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_is_hyper  = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;

    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_is_hyper  = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;

    //==========================================================================
    // phase1: count nnz(C(:,j)) for coarse tasks, scatter M for fine tasks
    //==========================================================================

    if (M == NULL)
    {

        //----------------------------------------------------------------------
        // C = A*B
        //----------------------------------------------------------------------

        if (A_is_sparse)
        {
            if (B_is_sparse)
            { 
                // both A and B are sparse
                GB_AxB_saxpy3_sym_ss (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_hyper)
            { 
                // A is sparse and B is hyper
                GB_AxB_saxpy3_sym_sh (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_bitmap)
            { 
                // A is sparse and B is bitmap
                GB_AxB_saxpy3_sym_sb (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is sparse and B is full
                GB_AxB_saxpy3_sym_sf (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
        else if (A_is_hyper)
        {
            if (B_is_sparse)
            { 
                // A is hyper and B is sparse
                GB_AxB_saxpy3_sym_hs (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_hyper)
            { 
                // both A and B are hyper
                GB_AxB_saxpy3_sym_hh (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_bitmap)
            { 
                // A is hyper and B is bitmap
                GB_AxB_saxpy3_sym_hb (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is hyper and B is full
                GB_AxB_saxpy3_sym_hf (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
        else if (A_is_bitmap)
        {
            // C=A*B where A is bitmap; B must be sparse/hyper
            if (B_is_sparse)
            { 
                // A is bitmap and B is sparse
                GB_AxB_saxpy3_sym_bs (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is bitmap and B is hyper
                ASSERT (B_is_hyper) ;
                GB_AxB_saxpy3_sym_bh (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
        else
        {
            // C=A*B where A is full; B must be sparse/hyper
            if (B_is_sparse)
            { 
                // A is full and B is sparse
                GB_AxB_saxpy3_sym_fs (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is full and B is hyper
                ASSERT (B_is_hyper) ;
                GB_AxB_saxpy3_sym_fh (C,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }

    }
    else if (!Mask_comp)
    {

        //----------------------------------------------------------------------
        // C<M> = A*B
        //----------------------------------------------------------------------

        if (A_is_sparse)
        {
            if (B_is_sparse)
            { 
                // both A and B are sparse
                GB_AxB_saxpy3_sym_mss (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_hyper)
            { 
                // A is sparse and B is hyper
                GB_AxB_saxpy3_sym_msh (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_bitmap)
            { 
                // A is sparse and B is bitmap
                GB_AxB_saxpy3_sym_msb (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is sparse and B is full
                GB_AxB_saxpy3_sym_msf (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
        else if (A_is_hyper)
        {
            if (B_is_sparse)
            { 
                // A is hyper and B is sparse
                GB_AxB_saxpy3_sym_mhs (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_hyper)
            { 
                // both A and B are hyper
                GB_AxB_saxpy3_sym_mhh (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_bitmap)
            { 
                // A is hyper and B is bitmap
                GB_AxB_saxpy3_sym_mhb (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is hyper and B is full
                GB_AxB_saxpy3_sym_mhf (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
        else if (A_is_bitmap)
        {
            if (B_is_sparse)
            { 
                // A is bitmap and B is sparse
                GB_AxB_saxpy3_sym_mbs (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_hyper)
            { 
                // A is bitmap and B is hyper
                GB_AxB_saxpy3_sym_mbh (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_bitmap)
            { 
                // both A and B are bitmap
                GB_AxB_saxpy3_sym_mbb (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is bitmap and B is full
                GB_AxB_saxpy3_sym_mbf (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
        else
        {
            if (B_is_sparse)
            { 
                // A is full and B is sparse
                GB_AxB_saxpy3_sym_mfs (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_hyper)
            { 
                // A is full and B is hyper
                GB_AxB_saxpy3_sym_mfh (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_bitmap)
            { 
                // A is full and B is bitmap
                GB_AxB_saxpy3_sym_mfb (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // both A and B are full
                GB_AxB_saxpy3_sym_mff (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C<!M> = A*B
        //----------------------------------------------------------------------

        if (A_is_sparse)
        {
            if (B_is_sparse)
            { 
                // both A and B are sparse
                GB_AxB_saxpy3_sym_nss (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_hyper)
            { 
                // A is sparse and B is hyper
                GB_AxB_saxpy3_sym_nsh (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_bitmap)
            { 
                // A is sparse and B is bitmap
                GB_AxB_saxpy3_sym_nsb (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is sparse and B is full
                GB_AxB_saxpy3_sym_nsf (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
        else if (A_is_hyper)
        {
            if (B_is_sparse)
            { 
                // A is hyper and B is sparse
                GB_AxB_saxpy3_sym_nhs (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_hyper)
            { 
                // both A and B are hyper
                GB_AxB_saxpy3_sym_nhh (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else if (B_is_bitmap)
            { 
                // A is hyper and B is bitmap
                GB_AxB_saxpy3_sym_nhb (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is hyper and B is full
                GB_AxB_saxpy3_sym_nhf (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
        else if (A_is_bitmap)
        {
            // C<!M>=A*B where A is bitmap; B must be sparse/hyper
            if (B_is_sparse)
            { 
                // A is bitmap and B is sparse
                GB_AxB_saxpy3_sym_nbs (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is bitmap and B is hyper
                ASSERT (B_is_hyper) ;
                GB_AxB_saxpy3_sym_nbh (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
        else
        {
            // C<!M>=A*B where A is full; B must be sparse/hyper
            if (B_is_sparse)
            { 
                // A is full and B is sparse
                GB_AxB_saxpy3_sym_nfs (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
            else
            { 
                // A is full and B is hyper
                ASSERT (B_is_hyper) ;
                GB_AxB_saxpy3_sym_nfh (C, M, Mask_struct, M_in_place,
                    A, B, SaxpyTasks, ntasks, nfine, nthreads) ;
            }
        }
    }
}

