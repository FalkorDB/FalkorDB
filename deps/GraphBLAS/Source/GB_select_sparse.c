//------------------------------------------------------------------------------
// GB_select_sparse:  select entries from a matrix (C is sparse/hypersparse)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

#include "GB_select.h"
#include "GB_ek_slice.h"
#ifndef GBCOMPACT
#include "GB_sel__include.h"
#endif
#include "GB_scalar_wrap.h"
#include "GB_stringify.h"

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_FREE_WORK (&Zp, Zp_size) ;           \
    GB_WERK_POP (Work, int64_t) ;           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;   \
    GB_FREE (&Cp, Cp_size) ;                \
    GB_FREE (&Ch, Ch_size) ;                \
    GB_FREE (&Ci, Ci_size) ;                \
    GB_FREE (&Cx, Cx_size) ;                \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_phybix_free (C) ;                    \
    GB_FREE_WORKSPACE ;                     \
}

GrB_Info GB_select_sparse
(
    GrB_Matrix C,
    const bool C_iso,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A,
    const int64_t ithunk,
    const GB_void *restrict athunk,
    const GB_void *restrict ythunk,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    GrB_Info info ;
    int64_t *restrict Zp = NULL ; size_t Zp_size = 0 ;
    GB_WERK_DECLARE (Work, int64_t) ;
    int64_t *restrict Wfirst = NULL ;
    int64_t *restrict Wlast = NULL ;
    int64_t *restrict Cp_kfirst = NULL ;
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;

    int64_t *restrict Cp = NULL ; size_t Cp_size = 0 ;
    int64_t *restrict Ch = NULL ; size_t Ch_size = 0 ;
    int64_t *restrict Ci = NULL ; size_t Ci_size = 0 ;
    GB_void *restrict Cx = NULL ; size_t Cx_size = 0 ;

    GB_Opcode opcode = op->opcode ;
    bool in_place_A = (C == NULL) ; // GrB_wait and GB_resize only
    const bool A_iso = A->iso ;
    const size_t asize = A->type->size ;
    const GB_Type_code acode = A->type->code ;

    //--------------------------------------------------------------------------
    // determine the max number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // get A: sparse, hypersparse, or full
    //--------------------------------------------------------------------------

    // the case when A is bitmap is always handled above by GB_select_bitmap
    ASSERT (!GB_IS_BITMAP (A)) ;

    int64_t *restrict Ap = A->p ; size_t Ap_size = A->p_size ;
    int64_t *restrict Ah = A->h ;
    int64_t *restrict Ai = A->i ; size_t Ai_size = A->i_size ;
    GB_void *restrict Ax = (GB_void *) A->x ; size_t Ax_size = A->x_size ;
    int64_t anvec = A->nvec ;
    bool A_jumbled = A->jumbled ;
    bool A_is_hyper = (Ah != NULL) ;
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;

    //--------------------------------------------------------------------------
    // allocate the new vector pointers of C
    //--------------------------------------------------------------------------

    int64_t cnz = 0 ;
    int64_t cplen = (avdim == 1) ? 1 : anvec ;

    Cp = GB_CALLOC (cplen+1, int64_t, &Cp_size) ;
    if (Cp == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // slice the entries for each task
    //--------------------------------------------------------------------------

    int A_ntasks, A_nthreads ;
    int64_t anz_held = GB_nnz_held (A) ;
    double work = 8*anvec + ((opcode == GB_DIAG_idxunop_code) ? 0 : anz_held) ;
    GB_SLICE_MATRIX_WORK (A, 8, work, anz_held) ;

    //--------------------------------------------------------------------------
    // allocate workspace for each task
    //--------------------------------------------------------------------------

    GB_WERK_PUSH (Work, 3*A_ntasks, int64_t) ;
    if (Work == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    Wfirst    = Work ;
    Wlast     = Work + A_ntasks ;
    Cp_kfirst = Work + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // allocate workspace for phase1
    //--------------------------------------------------------------------------

    // phase1 counts the number of live entries in each vector of A.  The
    // result is computed in Cp, where Cp [k] is the number of live entries in
    // the kth vector of A.  Zp [k] is the location of the A(i,k) entry, for
    // positional operators.

    bool op_is_positional = GB_IS_INDEXUNARYOP_CODE_POSITIONAL (opcode) ;
    if (op_is_positional)
    {
        // allocate Zp
        Zp = GB_MALLOC_WORK (cplen, int64_t, &Zp_size) ;
        if (Zp == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //==========================================================================
    // phase1: count the live entries in each column
    //==========================================================================

    info = GrB_NO_VALUE ;
    if (op_is_positional || opcode == GB_NONZOMBIE_idxunop_code)
    { 

        //----------------------------------------------------------------------
        // positional ops or nonzombie phase1 do not depend on the values
        //----------------------------------------------------------------------

        // no JIT worker needed for these operators
        info = GB_select_positional_phase1 (Zp, Cp, Wfirst, Wlast, A, ithunk,
            op, A_ek_slicing, A_ntasks, A_nthreads) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // entry selectors depend on the values in phase1
        //----------------------------------------------------------------------

        ASSERT (!A_iso || opcode == GB_USER_idxunop_code) ;
        ASSERT ((opcode >= GB_VALUENE_idxunop_code
             && opcode <= GB_VALUELE_idxunop_code)
             || (opcode == GB_USER_idxunop_code)) ;

        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        { 

            //------------------------------------------------------------------
            // via the factory kernel (includes user-defined ops)
            //------------------------------------------------------------------

            // define the worker for the switch factory
            #define GB_sel1(opname,aname) GB (_sel_phase1_ ## opname ## aname)
            #define GB_SEL_WORKER(opname,aname)                             \
            {                                                               \
                info = GB_sel1 (opname, aname) (Cp, Wfirst, Wlast, A,       \
                    ythunk, A_ek_slicing, A_ntasks, A_nthreads) ;           \
            }                                                               \
            break ;

            // launch the switch factory
            #include "GB_select_entry_factory.c"
            #undef  GB_SEL_WORKER
        }
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_select_phase1_jit (Cp, Wfirst, Wlast, C_iso, in_place_A,
                A, ythunk, op, flipij, A_ek_slicing, A_ntasks, A_nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            // generic entry selector, phase1
            info = GB_select_generic_phase1 (Cp, Wfirst, Wlast,
                A, flipij, ythunk, op, A_ek_slicing, A_ntasks, A_nthreads) ;
        }
    }

    //==========================================================================
    // phase1b: cumulative sum and allocate C
    //==========================================================================

    //--------------------------------------------------------------------------
    // cumulative sum of Cp and compute Cp_kfirst
    //--------------------------------------------------------------------------

    int64_t C_nvec_nonempty ;
    GB_ek_slice_merge2 (&C_nvec_nonempty, Cp_kfirst, Cp, anvec,
        Wfirst, Wlast, A_ek_slicing, A_ntasks, A_nthreads, Werk) ;

    //--------------------------------------------------------------------------
    // allocate new space for the compacted Ci and Cx
    //--------------------------------------------------------------------------

    cnz = Cp [anvec] ;
    cnz = GB_IMAX (cnz, 1) ;
    Ci = GB_MALLOC (cnz, int64_t, &Ci_size) ;
    // use calloc since C is sparse, not bitmap
    Cx = (GB_void *) GB_XALLOC (false, C_iso, cnz, asize, &Cx_size) ; // x:OK
    if (Ci == NULL || Cx == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // set the iso value of C
    //--------------------------------------------------------------------------

    if (C_iso)
    { 
        // The pattern of C is computed by the worker below.
        GB_select_iso (Cx, opcode, athunk, Ax, asize) ;
    }

    //==========================================================================
    // phase2: select the entries
    //==========================================================================

    info = GrB_NO_VALUE ;
    if (op_is_positional || (opcode == GB_NONZOMBIE_idxunop_code && A_iso))
    { 

        //----------------------------------------------------------------------
        // positional ops do not depend on the values
        //----------------------------------------------------------------------

        // no JIT worker needed for these operators
        info = GB_select_positional_phase2 (Ci, Cx, Zp, Cp, Cp_kfirst, A,
            flipij, ithunk, op, A_ek_slicing, A_ntasks, A_nthreads) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // entry selectors depend on the values in phase2
        //----------------------------------------------------------------------

        ASSERT (!A_iso || opcode == GB_USER_idxunop_code) ;
        ASSERT ((opcode >= GB_VALUENE_idxunop_code &&
                 opcode <= GB_VALUELE_idxunop_code)
             || (opcode == GB_NONZOMBIE_idxunop_code && !A_iso)
             || (opcode == GB_USER_idxunop_code)) ;

        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        { 

            //------------------------------------------------------------------
            // via the factory kernel
            //------------------------------------------------------------------

            // define the worker for the switch factory
            #define GB_SELECT_PHASE2
            #define GB_sel2(opname,aname) GB (_sel_phase2_ ## opname ## aname)
            #define GB_SEL_WORKER(opname,aname)                             \
            {                                                               \
                info = GB_sel2 (opname, aname) (Ci, Cx, Cp, Cp_kfirst, A,   \
                    ythunk, A_ek_slicing, A_ntasks, A_nthreads) ;           \
            }                                                               \
            break ;

            // launch the switch factory
            #include "GB_select_entry_factory.c"
        }
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_select_phase2_jit (Ci, C_iso ? NULL : Cx, Cp, C_iso,
                in_place_A, Cp_kfirst, A, flipij, ythunk, op, A_ek_slicing,
                A_ntasks, A_nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            // generic entry selector, phase2
            info = GB_select_generic_phase2 (Ci, C_iso ? NULL : Cx, Cp,
                Cp_kfirst, A, flipij, ythunk, op, A_ek_slicing, A_ntasks,
                A_nthreads) ;
        }
    }

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    //==========================================================================
    // finalize the result
    //==========================================================================

    if (in_place_A)
    {

        //----------------------------------------------------------------------
        // transplant Cp, Ci, Cx back into A
        //----------------------------------------------------------------------

        // TODO: this is not parallel: use GB_hyper_prune
        if (A->h != NULL && C_nvec_nonempty < anvec)
        {
            // prune empty vectors from Ah and Ap
            int64_t cnvec = 0 ;
            for (int64_t k = 0 ; k < anvec ; k++)
            {
                if (Cp [k] < Cp [k+1])
                { 
                    Ah [cnvec] = Ah [k] ;
                    Ap [cnvec] = Cp [k] ;
                    cnvec++ ;
                }
            }
            Ap [cnvec] = Cp [anvec] ;
            A->nvec = cnvec ;
            ASSERT (A->nvec == C_nvec_nonempty) ;
            GB_FREE (&Cp, Cp_size) ;
            // the A->Y hyper_hash is now invalid
            GB_hyper_hash_free (A) ;
        }
        else
        { 
            // free the old A->p and transplant in Cp as the new A->p
            GB_FREE (&Ap, Ap_size) ;
            A->p = Cp ; Cp = NULL ; A->p_size = Cp_size ;
            A->plen = cplen ;
        }

        ASSERT (Cp == NULL) ;

        GB_FREE (&Ai, Ai_size) ;
        GB_FREE (&Ax, Ax_size) ;
        A->i = Ci ; Ci = NULL ; A->i_size = Ci_size ;
        A->x = Cx ; Cx = NULL ; A->x_size = Cx_size ;
        A->nvec_nonempty = C_nvec_nonempty ;
        A->jumbled = A_jumbled ;        // A remains jumbled (in-place select)
        A->iso = C_iso ;                // OK: burble already done above
        A->nvals = A->p [A->nvec] ;

        // the NONZOMBIE opcode may have removed all zombies, but A->nzombie
        // is still nonzero.  It is set to zero in GB_wait.
        ASSERT_MATRIX_OK (A, "A output for GB_selector", GB_FLIP (GB0)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // create C and transplant Cp, Ch, Ci, Cx into C
        //----------------------------------------------------------------------

        int csparsity = (A_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE ;
        ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
        info = GB_new (&C, // sparse or hyper (from A), existing header
            A->type, avlen, avdim, GB_Ap_null, true,
            csparsity, A->hyper_switch, anvec) ;
        ASSERT (info == GrB_SUCCESS) ;

        if (A->h != NULL)
        {

            //------------------------------------------------------------------
            // A and C are hypersparse: copy non-empty vectors from Ah to Ch
            //------------------------------------------------------------------

            Ch = GB_MALLOC (anvec, int64_t, &Ch_size) ;
            if (Ch == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            // TODO: do in parallel: use GB_hyper_prune
            int64_t cnvec = 0 ;
            for (int64_t k = 0 ; k < anvec ; k++)
            {
                if (Cp [k] < Cp [k+1])
                { 
                    Ch [cnvec] = Ah [k] ;
                    Cp [cnvec] = Cp [k] ;
                    cnvec++ ;
                }
            }
            Cp [cnvec] = Cp [anvec] ;
            C->nvec = cnvec ;
            ASSERT (C->nvec == C_nvec_nonempty) ;
        }

        // note that C->Y is not yet constructed
        C->p = Cp ; Cp = NULL ; C->p_size = Cp_size ;
        C->h = Ch ; Ch = NULL ; C->h_size = Ch_size ;
        C->i = Ci ; Ci = NULL ; C->i_size = Ci_size ;
        C->x = Cx ; Cx = NULL ; C->x_size = Cx_size ;
        C->plen = cplen ;
        C->magic = GB_MAGIC ;
        C->nvec_nonempty = C_nvec_nonempty ;
        C->jumbled = A_jumbled ;    // C is jumbled if A is jumbled
        C->iso = C_iso ;            // OK: burble already done above
        C->nvals = C->p [C->nvec] ;

        ASSERT_MATRIX_OK (C, "C output for GB_selector", GB0) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

