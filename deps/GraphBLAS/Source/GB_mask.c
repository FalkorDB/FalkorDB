//------------------------------------------------------------------------------
// GB_mask: apply a mask: C<M> = Z
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = Z

// GB_mask is only called by GB_accum_mask.

// If M is NULL, C can have any sparsity.  Otherwise, if M is present then
// C is sparse or hypersparse; if bitmap or full, GB_subassign is used instead.

// Nearly all GraphBLAS operations take a mask, which controls how the result
// of the computations, Z, are copied into the result matrix C.  The following
// working script, GB_spec_mask.m, defines how this is done.  In the
// comments, C(i,j) is shorthand for the index i in the jth vector, and
// likewise for M, Z, and R.  If the matrices are all CSC, then this is row i
// and column j.  If the matrices are all CSR, then it is row j and column i.

/*

    function R = GB_spec_mask (C, M, Z, C_replace, Mask_comp,identity)
    %GB_SPEC_MASK: an implementation of GB_mask
    %
    % Computes C<M> = Z, in GraphBLAS notation.
    %
    % Usage:
    % C = GB_spec_mask (C, M, Z, C_replace, Mask_comp, identity)
    %
    % C and Z: matrices of the same size.
    %
    % optional inputs:
    % M: if empty or not present, M = ones (size (C))
    % C_replace: set C to zero first. Default is false.
    % Mask_comp: use ~M instead of M. Default is false.
    % identity: the additive identity of the semiring.  Default is zero.
    %   This is only needed because the GB_spec_* routines operate on dense
    %   matrices, and thus they need to know the value of the implicit 'zero'.
    %
    % This method operates on both plain matrices and on structs with
    % matrix, pattern, and class components.

    if (nargin < 6)
        identity = 0 ;
    end
    if (nargin < 5)
        Mask_comp = false ;
    end
    if (nargin < 4)
        C_replace = false ;
    end

    if (isstruct (C))
        % apply the mask to both the matrix and the pattern
        R.matrix  = GB_spec_mask (C.matrix,  M, Z.matrix,  C_replace, ...
            Mask_comp, identity) ;
        R.pattern = GB_spec_mask (C.pattern, M, Z.pattern, C_replace, ...
            Mask_comp, false) ;
        R.class = C.class ;
        return
    end

    if (~isequal (size (C), size (Z)))
        error ('C and Z must have the same size') ;
    end
    if (~isempty (M))
        if (~isequal (size (C), size (M)))
            error ('C and M must have the same size') ;
        end
    end

    % replace C if requested
    if (C_replace)
        C (:,:) = identity ;
    end

    if (isempty (M))
        % in GraphBLAS, this means M is NULL;
        % implicitly, M = ones (size (C))
        if (~Mask_comp)
            R = Z ;
        else
            % note that Z need never have been computed
            R = C ;
        end
    else
        % form the Boolean mask. For GraphBLAS, this does the
        % right thing and ignores explicit zeros in M.
        M = (M ~= 0) ;
        if (~Mask_comp)
            % R will equal C where M is false
            R = C ;
            % overwrite R with Z where M is true
            R (M) = Z (M) ;
        else
            % M is complemented
            % R will equal Z where M is false
            R = Z ;
            % overwrite R with C where M is true
            R (M) = C (M) ;
        end
    end

*/

#define GB_FREE_ALL                     \
{                                       \
    GB_Matrix_free (Zhandle) ;          \
    GB_Matrix_free (&C0) ;              \
    GB_Matrix_free (&R) ;               \
}

#include "GB_mask.h"

//------------------------------------------------------------------------------

GrB_Info GB_mask                // C<M> = Z
(
    GrB_Matrix C_result,        // both input C and result matrix
    const GrB_Matrix M,         // optional mask matrix, can be NULL
    GrB_Matrix *Zhandle,        // Z = results of computation, perhaps shallow.
                                // Z is freed when done.
    const bool C_replace,       // true if clear(C) to be done first
    const bool Mask_comp,       // true if M is to be complemented
    const bool Mask_struct,     // if true, use the only structure of M
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C_result may be aliased with M
    ASSERT_MATRIX_OK (C_result, "C_result for GB_mask", GB0) ;
    ASSERT (!C_result->static_header) ;
    // C may be cleared anyway, without the need for finishing it
    ASSERT (GB_ZOMBIES_OK (C_result)) ;
    ASSERT (GB_JUMBLED_OK (C_result)) ;
    ASSERT (GB_PENDING_OK (C_result)) ;

    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_mask", GB0) ;
    // M may have zombies and pending tuples
    ASSERT (GB_PENDING_OK (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (GB_ZOMBIES_OK (M)) ;

    // Z has the same type as C_result, with no zombies or pending tuples
    ASSERT (Zhandle != NULL) ;
    GrB_Matrix Z = *Zhandle ;
    ASSERT_MATRIX_OK (Z, "Z for GB_mask", GB0) ;
    ASSERT (!GB_PENDING (Z)) ;
    ASSERT (GB_JUMBLED_OK (Z)) ;
    ASSERT (!GB_ZOMBIES (Z)) ;
    ASSERT (Z->type == C_result->type) ;
    // Z and C_result are never aliased. C_result and M might be.
    ASSERT (Z != C_result) ;
    // Z and C_result must have the same format and dimensions
    ASSERT (C_result->vlen == Z->vlen) ;
    ASSERT (C_result->vdim == Z->vdim) ;

    // M must be compatible with C_result
    ASSERT_OK (GB_Mask_compatible (M, Mask_struct, C_result, 0, 0, Werk)) ;

    GrB_Info info = GrB_SUCCESS ;
    GrB_Matrix C = NULL, C0 = NULL, R = NULL ;
    struct GB_Matrix_opaque C0_header, R_header ;

    //--------------------------------------------------------------------------
    // apply the mask
    //--------------------------------------------------------------------------

    if (M == NULL)
    {

        //----------------------------------------------------------------------
        // there is no mask (implicitly M(i,j)=1 for all i and j)
        //----------------------------------------------------------------------

        // Any pending work on C is abandoned (zombies and/or pending tuples).
        // C and Z can have any sparsity, including bitmap or full.

        if (!Mask_comp)
        { 

            //------------------------------------------------------------------
            // mask is not complemented: this is the default
            //------------------------------------------------------------------

            // C_result = Z, but make sure a deep copy is made as needed.  It is
            // possible that Z is a shallow copy of another matrix.
            // Z is freed by GB_transplant_conform.
            ASSERT (!C_result->p_shallow) ;
            ASSERT (!C_result->h_shallow) ;

            // transplant Z into C_result and conform to desired hypersparsity
            return (GB_transplant_conform (C_result, C_result->type, Zhandle,
                Werk)) ;
        }
        else
        {

            //------------------------------------------------------------------
            // an empty mask is complemented: Z is ignored
            //------------------------------------------------------------------

            // Z is ignored, and can even be NULL.  The method that calls
            // GB_mask can short circuit its computation, ignore accum, and
            // apply the mask immediately, and then return to its caller.
            // This done by the GB_RETURN_IF_QUICK_MASK macro.

            // In the current version, this work is always done by the
            // GB_RETURN_IF_QUICK_MASK macro, and GB_mask is no longer called
            // with an empty complemented mask.  The following is thus dead
            // code.  It is kept here in case this function is called to handle
            // this case in a future version.

            ASSERT (GB_DEAD_CODE) ;    // the following is no longer used

            // free Z if it exists (this is OK if Zhandle is NULL)
            GB_Matrix_free (Zhandle) ;

            if (C_replace)
            {
                // C_result = 0
                return (GB_clear (C_result, Werk)) ;
            }
            else
            {
                // nothing happens
                return (GrB_SUCCESS) ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // the mask is present
        //----------------------------------------------------------------------

        // delete any lingering zombies and assemble any pending tuples
        GB_MATRIX_WAIT (M) ;        // also sort M if jumbled
        GB_MATRIX_WAIT (Z) ;        // also sort Z if jumbled

        // R has the same CSR/CSC format as C_result.  It is hypersparse if
        // both C and Z are hypersparse.

        bool R_is_csc = C_result->is_csc ;
        int64_t vdim = C_result->vdim ;
        int64_t vlen = C_result->vlen ;

        if (C_replace)
        {
            if (GB_any_aliased (C_result, M))
            { 
                // C_result and M are aliased.  This is OK, unless C_replace is
                // true.  In this case, M must be left unchanged but C_result
                // must be cleared.  To resolve this, a new matrix C0 is
                // created, which is what C_result would look like if cleared.
                // C_result is left unchanged since changing it would change M.
                // The C0 matrix is created as hypersparse.
                // set C0->iso = false  OK
                GB_CLEAR_STATIC_HEADER (C0, &C0_header) ;
                GB_OK (GB_new_bix (&C0, // sparse or hyper, existing header
                    C_result->type, vlen, vdim, GB_Ap_calloc, R_is_csc,
                    GxB_HYPERSPARSE, true, C_result->hyper_switch, 0, 0,
                    true, false)) ;
                C = C0 ;
                ASSERT (C->static_header || GBNSTATIC) ;
            }
            else
            { 
                // Clear all entries from C_result, and ensure C is hypersparse
                // by temporarily changing the sparsity control
                int save = C_result->sparsity_control ;     // save control
                C_result->sparsity_control = GxB_HYPERSPARSE ;
                GB_OK (GB_clear (C_result, Werk)) ;
                C_result->sparsity_control = save ;         // restore control
                C = C_result ;  // C must have a dynamic header
                ASSERT (!C->static_header) ;
            }
            // C has been cleared, so it has no zombies or pending tuples
        }
        else
        { 
            // C has already been finished if C_replace is false, via the
            // GB_MATRIX_WAIT (C) in GB_accum_mask.
            C = C_result ;
            ASSERT (!C->static_header) ;
        }

        // C cannot be bitmap or full for GB_masker
        ASSERT (!GB_IS_BITMAP (C)) ;
        ASSERT (!GB_IS_FULL (C)) ;

        // no more zombies or pending tuples in M or C
        ASSERT (!GB_PENDING (M)) ;
        ASSERT (!GB_JUMBLED (M)) ;
        ASSERT (!GB_ZOMBIES (M)) ;
        ASSERT (!GB_PENDING (C)) ;
        ASSERT (!GB_JUMBLED (C)) ;
        ASSERT (!GB_ZOMBIES (C)) ;

        // continue with C, do not use C_result until the end since it may be
        // aliased with M.

        //----------------------------------------------------------------------
        // R = masker (C, M, Z):  compute C<M>=Z, placing results in R
        //----------------------------------------------------------------------

        GB_CLEAR_STATIC_HEADER (R, &R_header) ;
        GB_OK (GB_masker (R, R_is_csc, M, Mask_comp, Mask_struct, C, Z,
            Werk)) ;

        //----------------------------------------------------------------------
        // free temporary matrices Z and C0
        //----------------------------------------------------------------------

        GB_Matrix_free (Zhandle) ;
        GB_Matrix_free (&C0) ;

        //----------------------------------------------------------------------
        // transplant the result, conform, and free R
        //----------------------------------------------------------------------

        // finished using the mask M, so it is now safe to modify C_result,
        // even if C_result and M are aliased

        return (GB_transplant_conform (C_result, R->type, &R, Werk)) ;
    }
}

