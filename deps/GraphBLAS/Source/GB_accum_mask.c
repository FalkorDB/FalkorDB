//------------------------------------------------------------------------------
// GB_accum_mask: accumulate results via the mask and accum operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,T)

// The primary computation of a GraphBLAS operations is done, and the results
// are in the T matrix.  The T matrix is then used to modify C, via the accum
// operator and the mask matrix M.

// The results are first accumulated into Z via the accum operator.

// Let Z = accum (C,T) if accum is present, or Z = T otherwise.
// In either case, the type of Z is the same as the C->type defined on input.
// If accum is present, T is typecast into the type of the y input to accum.
// If accum is not present, T is typecast into the same type as C.

// If the function z = accum(x,y) is present, then it defines how the existing
// values of C are used to accumulate T into Z.  If both T(i,j) and C(i,j) are
// present in the pattern, then Z(i,j) = accum (C(i,j), T(i,j)).  Otherwise,
// accum is not used: If C(i,j) is present but not T(i,j), then
// Z(i,j)=C(i,j).  If C(i,j) is not present but T(i,j) is present, then
// Z(i,j)=T(i,j).  The pattern of Z = accum(C,T) is the union of C and T.

// The Z = accum (C,T) phase is mimiced by the GB_spec_accum.m script.

// The next step is C<M> = Z.

// This denotes how the matrix Z is written into C, under the control of the
// mask (or !M if Mask_comp is true), and the C_replace flag (which
// indicates that C should be set to zero first.  This is C<M>=Z in
// GraphBLAS notation.  See GB_mask.c, or GB_spec_mask.m for a script
// that describes this step.

// If M is not present, C = Z is returned. Otherwise, M defines what
// values of C are modified. If M(i,j) is present and nonzero, then
// C(i,j)=Z(i,j) is done.  Otherwise, C(i,j) is left unchanged.

// The descriptor affects how C and M are handled.  If the descriptor is
// NULL, defaults are used.

#define GB_FREE_ALL                 \
{                                   \
    GB_Matrix_free (Thandle) ;      \
    GB_Matrix_free (&MT) ;          \
    GB_Matrix_free (&Z) ;           \
}

#include "GB_subassign.h"
#include "GB_add.h"
#include "GB_mask.h"
#include "GB_transpose.h"
#include "GB_accum_mask.h"
#include "GB_bitmap_assign.h"
#include "GB_unused.h"

/* -----------------------------------------------------------------------------

    function Z = GB_spec_accum (accum, C, T, identity)
    %GB_SPEC_ACCUM: a mimic of the Z=accum(C,T) operation in GraphBLAS
    %
    % Z = GB_spec_accum (accum, C, T, identity)
    %
    % Apply accum binary operator to the input C and the intermediate result T.
    %

    % get the operator; default is class(C) if class is not present
    [opname opclass] = GB_spec_operator (accum, C.class) ;

    if (nargin < 4)
        identity = 0 ;
    end

    % initialize the matrix Z, same size and class as C
    [nrows ncols] = size (C.matrix) ;
    Z.matrix  = zeros (nrows, ncols, C.class) ;
    Z.matrix (:,:) = identity ;
    Z.pattern = false (nrows, ncols) ;
    Z.class = C.class ;

    if (isempty (opname))

        % Z = T, casting into the class of C
        Z.matrix  = GB_mex_cast (T.matrix, C.class) ;
        Z.pattern = T.pattern ;

    else

        % Z = accum (C,T)

        % apply the operator to entries in the intersection of C and T
        p = T.pattern & C.pattern ;
        % first cast the entries into the class of the operator
        % note that in the spec, all three domains z=op(x,y) can be different
        % here they are assumed to all be the same
        c = GB_mex_cast (C.matrix (p), opclass) ;
        t = GB_mex_cast (T.matrix (p), opclass) ;
        z = GB_spec_op (accum, c, t) ;
        % cast the result z from opclass into the class of C
        Z.matrix (p) = GB_mex_cast (z, C.class) ;

        % copy entries in C but not in T, into the result Z, no typecasting
        p = C.pattern & ~T.pattern ;
        Z.matrix (p) = C.matrix (p) ;

        % cast entries in T but not in C, into the result Z
        p = T.pattern & ~C.pattern ;
        Z.matrix (p) = GB_mex_cast (T.matrix (p), C.class) ;

        % the pattern of Z is the union of both T and C
        Z.pattern = C.pattern | T.pattern ;

    end

----------------------------------------------------------------------------- */

//------------------------------------------------------------------------------
// GB_accum_mask
//------------------------------------------------------------------------------

GrB_Info GB_accum_mask          // C<M> = accum (C,T)
(
    GrB_Matrix C,               // input/output matrix for results
    const GrB_Matrix M_in,      // optional mask for C, unused if NULL
    const GrB_Matrix MT_in,     // MT=M' if computed already in the caller
    const GrB_BinaryOp accum,   // optional accum for Z=accum(C,results)
    GrB_Matrix *Thandle,        // results of computation, freed when done
    const bool C_replace,       // if true, clear C first
    const bool Mask_comp,       // if true, complement the mask
    const bool Mask_struct,     // if true, use the only structure of M
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M_in

    ASSERT (Thandle != NULL) ;
    GrB_Info info ;
    GrB_Matrix T = *Thandle ;
    struct GB_Matrix_opaque MT_header, Z_header ;
    GrB_Matrix MT = NULL, Z = NULL ;
    GrB_Matrix M = M_in ;

    ASSERT_MATRIX_OK (C, "C input for C<M>=accum(C,T)", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_accum_mask", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (MT_in, "MT_in for GB_accum_mask", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_accum_mask", GB0) ;
    ASSERT (!GB_OP_IS_POSITIONAL (accum)) ;

    // pending work in C may be abandoned, or it might not need to be
    // finished if GB_subassign is used, so it is not finished here.
    ASSERT (GB_PENDING_OK (C)) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;

    ASSERT (GB_PENDING_OK (M)) ;
    ASSERT (GB_ZOMBIES_OK (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;

    // pending work in T will be finished now
    ASSERT (GB_PENDING_OK (T)) ;
    ASSERT (GB_ZOMBIES_OK (T)) ;
    ASSERT (GB_JUMBLED_OK (T)) ;
    ASSERT_MATRIX_OK (T, "[T = results of computation]", GB0) ;

    //--------------------------------------------------------------------------
    // remove zombies and pending tuples from T, but leave it jumbled
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (T) ;
    ASSERT (!GB_PENDING (T)) ;
    ASSERT (!GB_ZOMBIES (T)) ;
    ASSERT (GB_JUMBLED_OK (T)) ;

    //--------------------------------------------------------------------------
    // ensure M and T have the same CSR/CSC format as C
    //--------------------------------------------------------------------------

    bool T_transposed = false ;
    bool M_transposed = false ;

    if (C->is_csc != T->is_csc)
    { 
        // T can be jumbled.
        ASSERT (GB_JUMBLED_OK (T)) ;
        GB_OK (GB_transpose_in_place (T, C->is_csc, Werk)) ;
        T_transposed = true ;
        ASSERT (GB_JUMBLED_OK (T)) ;
        ASSERT_MATRIX_OK (T, "[T = transposed]", GB0) ;
    }

    if (M != NULL && C->is_csc != M->is_csc)
    {
        // M and C have different CSR/CSC formats.  This implies 
        // that C and M are not aliased.

        // MT = M' to conform M to the same CSR/CSC format as C.
        if (MT_in == NULL)
        { 
            // remove zombies and pending tuples from M.  M can be jumbled.
            GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (M) ;
            ASSERT (GB_JUMBLED_OK (M)) ;
            GB_CLEAR_STATIC_HEADER (MT, &MT_header) ;
            GB_OK (GB_transpose_cast (MT, GrB_BOOL, C->is_csc, M, Mask_struct,
                Werk)) ;
            ASSERT (MT->static_header || GBNSTATIC) ;
            // use the transpose mask
            M = MT ;
            ASSERT (GB_JUMBLED_OK (M)) ;
            M_transposed = true ;
        }
        else
        { 
            // Use the transpose mask passed in by the caller.
            // It is the right vlen-by-vdim dimension, but its
            // CSR/CSC format is ignored.
            M = MT_in ;
        }
    }

    // T and M now conform to the dimensions and CSR/CSC format of C
    ASSERT (C->vlen == T->vlen && C->vdim == T->vdim) ;
    ASSERT (C->is_csc == T->is_csc) ;
    ASSERT (M == NULL || (C->vlen == M->vlen && C->vdim == M->vdim)) ;
    ASSERT (M == NULL || (C->is_csc == M->is_csc)) ;
    ASSERT (!GB_PENDING (T)) ;
    ASSERT (!GB_ZOMBIES (T)) ;

    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (GB_JUMBLED_OK (T)) ;

    //--------------------------------------------------------------------------
    // decide on the method
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz (C) ;              // includes live entries and zombies
    int64_t cnpending = GB_Pending_n (C) ;  // # pending tuples in C
    int64_t tnz = GB_nnz (T) ;

    // Use subassign for the accum/mask step if either M or accum is present
    // (or both), and if the update is small compared to the size of C.
    // tnz+cnpending is an upper bound on the number of pending tuples in C
    // after the accum/mask step with subassign.  If this is small (< nnz(C)),
    // then use subassign.  It will be fast when T is very sparse and C has
    // many nonzeros.  If the # of pending tuples in C is growing, however,
    // then it would be better to finish the work now, and leave C completed.
    // In this case, GB_transplant if no accum or GB_add with accum, and
    // GB_mask are used for the accum/mask step.

    // If there is no mask M, and no accum, then C=T is fast (just
    // GB_transplant for Z=T and GB_transplant_conform in GB_mask for C=Z).
    // So in this case, GB_subassign takes more work.

    if (GB_any_aliased (C, M)) GBURBLE ("(C aliased with M) ") ;
    if (GB_any_aliased (C, T)) GBURBLE ("(C aliased with T) ") ;

    bool use_subassign = false ;

    if (M != NULL || accum != NULL)
    {
        if (GB_IS_BITMAP (C) || GB_IS_FULL (C))
        { 
            // always use GB_subassign if C is bitmap or full and M and/or
            // accum is present.  No zombies or pending tuples are introduced
            // into C, and C is modified in-place, so GB_subassign is very
            // efficient in this case.
            use_subassign = true ;
        }
        else
        { 
            // C is sparse or hypersparse (at least for now, before any wait on
            // C): use GB_subassign if the update is small (resuling in a small
            // number of pending tuples), and if C is not aliased with M or T.
            use_subassign = (tnz + cnpending <= cnz)
                && !GB_any_aliased (C, M) && !GB_any_aliased (C, T) ;
        }
    }

    bool use_transplant = (!use_subassign)
        && (accum == NULL || (cnz + cnpending) == 0) ;

    if (!use_subassign && (!use_transplant || (M != NULL && !C_replace)))
    {
        // GB_accum_mask will be used instead of GB_subassign, or so it
        // appears.  GB_subassign does not require the pending work in C to be
        // finished, but GB_accum_mask does in most cases.  Finish the work on
        // C now.  This may change C to bitmap/full, so recheck the bitmap/full
        // condition on C after doing the GB_MATRIX_WAIT (C).
        GB_MATRIX_WAIT (C) ;
        if (GB_IS_BITMAP (C) || GB_IS_FULL (C))
        { 
            // See Test/test182 for a test that triggers this condition.
            // GB_MATRIX_WAIT (C) has changed C from sparse/hyper to
            // bitmap/full.  GB_mask does not handle the case where M is
            // present, C_replace is false, and C is bitmap/full, so switch to
            // GB_subassign.
            use_subassign = true ;
        }
    }

    // use_subassign has been reconsidered and the pending work on C may now
    // be finished, which changes cnz and cnpending.  Recompute use_transplant.
    cnz = GB_nnz (C) ;              // includes live entries and zombies
    cnpending = GB_Pending_n (C) ;  // # pending tuples in C
    use_transplant = (!use_subassign)
        && (accum == NULL || (cnz + cnpending) == 0) ;

    // burble the decision on which method to use
    if (!use_transplant)
    { 
        GBURBLE ("(C%s%s=Z via %s%s%s) ",
            ((M == NULL) ? "" : ((Mask_comp) ? "<!M>" : "<M>")),
            ((accum == NULL) ? "" : "+"),
            ((use_subassign) ? "assign" : "add"),
            (M_transposed ? "(M transposed)" : ""),
            (T_transposed ? "(result transposed)" : "")) ;
    }

    //--------------------------------------------------------------------------
    // apply the accumulator and the mask
    //--------------------------------------------------------------------------

    if (use_subassign)
    { 

        //----------------------------------------------------------------------
        // C(:,:)<M> = accum (C(:,:),T) via GB_subassign
        //----------------------------------------------------------------------

        GB_OK (GB_subassign (C, C_replace, M, Mask_comp, Mask_struct,
            false, accum, T, false, GrB_ALL, 0, GrB_ALL, 0,
            false, NULL, GB_ignore_code, Werk)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C<M> = accum (C,T) via GB_transplant or GB_add, and GB_mask
        //----------------------------------------------------------------------

        // see GB_spec_accum.m for a description of this step.  If C is empty,
        // then the accumulator can be ignored.

        GB_CLEAR_STATIC_HEADER (Z, &Z_header) ;

        if (use_transplant)
        { 

            //------------------------------------------------------------------
            // Z = (ctype) T
            //------------------------------------------------------------------

            // GB_new allocates just the header for Z; the rest can be
            // allocated by the transplant if needed.  Z has the same
            // hypersparsity as T.

            info = GB_new (&Z, // sparse or hyper, existing header
                C->type, C->vlen, C->vdim, GB_Ap_null, C->is_csc,
                GB_sparsity (T), T->hyper_switch, T->plen) ;
            GB_OK (info) ;

            // Transplant T into Z, typecasting if needed, and free T.  This
            // may need to do a deep copy if T is shallow.  T is always freed
            // by GB_transplant.

            // Z and T have same vlen, vdim, is_csc, hypersparsity
            GB_OK (GB_transplant (Z, C->type, Thandle, Werk)) ;

        }
        else
        { 

            //------------------------------------------------------------------
            // Z = (ctype) accum (C,T)
            //------------------------------------------------------------------

            // GB_add_sparsity needs the final sparsity pattern of C and T,
            // so wait on C and T first.
            GB_MATRIX_WAIT (C) ;
            GB_MATRIX_WAIT (T) ;

            bool apply_mask ;
            int Z_sparsity = GB_add_sparsity (&apply_mask, M, Mask_struct,
                Mask_comp, C, T) ;

            // whether or not GB_add chooses to exploit the mask, it must still
            // be used in GB_mask, below.  So ignore the mask_applied return
            // flag from GB_add.
            bool ignore ;
            GB_OK (GB_add (Z, C->type, C->is_csc, (apply_mask) ? M : NULL,
                Mask_struct, Mask_comp, &ignore, C, T, false, NULL, NULL,
                accum, false, Werk)) ;
            GB_Matrix_free (Thandle) ;
        }

        // T has been transplanted into Z or freed after Z=C+T
        ASSERT (*Thandle == NULL ||
               (*Thandle != NULL && ((*Thandle)->static_header || GBNSTATIC))) ;

        // C and Z have the same type
        ASSERT_MATRIX_OK (Z, "Z in accum_mask", GB0) ;
        ASSERT (Z->type == C->type) ;

        //----------------------------------------------------------------------
        // apply the mask (C<M>=Z) and free Z
        //----------------------------------------------------------------------

        ASSERT_MATRIX_OK (C, "C<M>=Z input", GB0) ;
        GB_OK (GB_mask (C, M, &Z, C_replace, Mask_comp, Mask_struct, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "C<M>=accum(C,T)", GB0) ;
    return (GB_block (C, Werk)) ;
}

