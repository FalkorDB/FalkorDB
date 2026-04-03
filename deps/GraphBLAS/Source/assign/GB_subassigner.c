//------------------------------------------------------------------------------
// GB_subassigner: C(I,J)<#M> = accum (C(I,J), A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Submatrix assignment: C(I,J)<M> = A, or accum (C (I,J), A), no transpose

// All assignment operations rely on this function, including the GrB_*_assign
// operations in the spec, and the GxB_*_subassign operations that are a
// SuiteSparse:GraphBLAS extension to the spec:

// GrB_Matrix_assign,
// GrB_Matrix_assign_TYPE,
// GrB_Vector_assign,
// GrB_Vector_assign_TYPE,
// GrB_Row_assign,
// GrB_Col_assign

// GxB_Matrix_subassign,
// GxB_Matrix_subassign_TYPE,
// GxB_Vector_subassign,
// GxB_Vector_subassign_TYPE,
// GxB_Row_subassign,
// GxB_Col_subassign

// This function handles the accumulator, and the mask M, and the C_replace
// option itself, without relying on GB_accum_mask or GB_mask.  The mask M has
// the same size as C(I,J) and A.  M(0,0) governs how A(0,0) is assigned
// into C(I[0],J[0]).  This is how GxB_subassign operates.  For GrB_assign, the
// mask M in this function is the SubMask, constructed via SubMask=M(I,J).

// No transposed case is handled.  This function is also agnostic about the
// CSR/CSC format of C, A, and M.  The A matrix must have A->vlen == nI and
// A->vdim == nJ (except for scalar expansion, in which case A is NULL).  The
// mask M must be the same size as A, if present.

// Any or all of the C, M, and/or A matrices may be hypersparse or standard
// non-hypersparse.  Some methods can operate on full and/or bitmap matrices;
// see GB_subassigner_method, which checks these conditions.

// C is operated on in-place and thus cannot be aliased with the inputs A or M.

// Since the pattern of C isn't reallocated here, and entries do not move in
// position, C->p, C->h, C->nvec, and C->nvec_nonempty are not modified.  C->x
// and C->i can be modified, but only one entry at a time.  No entries are
// shifted.  C->i can be changed by turning an entry into a zombie, or by
// bringing a zombie back to life, but no entry in C->i moves in position, and
// the underlying indices in C->i do not change otherwise.  C->b can be
// modified for a C bitmap.

// C->x and C->iso have already been computed if C is iso on output, by
// GB_assign_prep, so if C->iso is true, there is no numeric work to do.

#include "assign/GB_subassign.h"
#include "assign/GB_subassign_methods.h"
#include "assign/GB_subassign_dense.h"
#include "assign/GB_bitmap_assign.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_phybix_free (C) ;

GrB_Info GB_subassigner             // C(I,J)<#M> = A or accum (C (I,J), A)
(
    // input/output
    GrB_Matrix C,                   // input/output matrix for results
    // input
    const int subassign_method,
    const bool C_replace,           // C matrix descriptor
    const GrB_Matrix M,             // optional mask for C(I,J), unused if NULL
    const bool Mask_comp,           // mask descriptor
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),A)
    const GrB_Matrix A,             // input matrix (NULL for scalar expansion)
    const void *I,                  // I index list
    const bool I_is_32,
    const int64_t ni,               // number of indices
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,                  // J index list
    const bool J_is_32,
    const int64_t nj,               // number of column indices
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const bool scalar_expansion,    // if true, expand scalar to A
    const void *scalar,             // scalar to be expanded
    const GrB_Type scalar_type,     // type code of scalar to expand
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C input for subassigner", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    //--------------------------------------------------------------------------
    // methods that rely on C and A being dense assume they are not jumbled
    //--------------------------------------------------------------------------

    if (GB_is_dense (A))
    { 
        // methods that rely on A being dense assume A is not jumbled
        GB_UNJUMBLE (A) ;
    }

    if (GB_is_dense (C) && !GB_PENDING_OR_ZOMBIES (C) && !GB_IS_BITMAP (C))
    { 
        // C is dense or full (as-if): if sparse, make sure it's unjumbled
        GB_UNJUMBLE (C) ;
    }

    GBURBLE ("(pending: " GBd ") ", GB_Pending_n (C)) ;

    //==========================================================================
    // submatrix assignment C(I,J)<M> = accum (C(I,J),A): meta-algorithm
    //==========================================================================

    // There are up to 128 combinations of options, but not all must be
    // implemented, because they are either identical to another method
    // (C_replace is effectively false if M=NULL and Mask_comp=false), or they
    // are not used (the last option, whether or not S is constructed, is
    // determined here; it is not a user input).  The first 5 options are
    // determined by the input.  The table below has been pruned to remove
    // combinations that are not used, or equivalent to other entries in the
    // table.  Only 22 unique combinations of the 128 combinations are needed,
    // with additional special cases when C(:,:) is dense.

    //      M           present or NULL
    //      Mask_comp   true or false
    //      Mask_struct structural or valued mask
    //      C_replace   true or false
    //      accum       present or NULL
    //      A           scalar (x) or matrix (A)
    //      S           constructed or not 

    // C(I,J)<(M,comp,repl)> ( = , += ) (A, scalar), (with or without S);
    // I and J can be anything for any of these methods (":", colon, or list).

    // See the "No work to do..." comment above:
    // If M is not present, Mask_comp true, C_replace false: no work to do.
    // If M is not present, Mask_comp true, C_replace true: use Method 00
    // If M is not present, Mask_comp false:  C_replace is now false.

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============

        //  -   -   x   -   -   -       21:  C = x, no S, C anything
        //  -   -   x   -   A   -       24:  C = A, no S, C and A anything
        //  -   -   -   +   -   -       22:  C += x, no S, C dense
        //  -   -   -   +   A   -       23:  C += A, no S, C dense

        //  -   -   -   -   -   S       01:  C(I,J) = x, with S
        //  -   -   -   -   A   S       02:  C(I,J) = A, with S
        //  -   -   -   +   -   S       03:  C(I,J) += x, with S
        //  -   -   -   +   A   S       04:  C(I,J) += A, with S
        //  -   -   r                        uses methods 01, 02, 03, 04
        //  -   c   -                        no work to do
        //  -   c   r           S       00:  C(I,J)<!,repl> = empty, with S

        //  M   -   -   -   -   -       05d: C<M> = x, no S, C dense
        //  M   -   -   -   -   -       05e: C<M,s> = x, no S, C empty
        //  M   -   -   -   -   -       05f: C<C,s> = x, no S, C == M
        //  M   -   -   -   -   -       05:  C(I,J)<M> = x, no S
        //  A   -   -   -   A   -       06d: C<A> = A, no S, C dense
        //  M   -   -   -   A   -       25:  C<M,s> = A, A dense, C empty
        //  M   -   -   -   A   -       06n: C(I,J)<M> = A, no S
        //  M   -   -   -   A   S       06s: C(I,J)<M> = A, with S
        //  M   -   -   +   -   -       07:  C(I,J)<M> += x, no S
        //  C   -   -   +   A   -       27:  C<C,struct> += A
        //  M   -   -   +   A   -       08n: C(I,J)<M> += A, no S
        //  M   -   -   +   A   S       08s: C(I,J)<M> += A, with S
        //  M   -   r   -   -   S       09:  C(I,J)<M,repl> = x, with S
        //  M   -   r   -   A   S       10:  C(I,J)<M,repl> = A, with S
        //  M   -   r   +   -   S       11:  C(I,J)<M,repl> += x, with S
        //  M   -   r   +   A   S       12:  C(I,J)<M,repl> += A, with S

        //  M   c   -   -   -   S       13:  C(I,J)<!M> = x, with S
        //  M   c   -   -   A   S       14:  C(I,J)<!M> = A, with S
        //  M   c   -   +   -   S       15:  C(I,J)<!M> += x, with S
        //  M   c   -   +   A   S       16:  C(I,J)<!M> += A, with S
        //  M   c   r   -   -   S       17:  C(I,J)<!M,repl> = x, with S
        //  M   c   r   -   A   S       18:  C(I,J)<!M,repl> = A, with S
        //  M   c   r   +   -   S       19:  C(I,J)<!M,repl> += x, with S
        //  M   c   r   +   A   S       20:  C(I,J)<!M,repl> += A, with S


        //----------------------------------------------------------------------
        // FUTURE::: 8 simpler cases when I and J are ":" (S not needed):
        //----------------------------------------------------------------------

        // These methods could all tolerate C==M and C==A aliasing, assuming no
        // binary search or if the binary search of C==M can be done with
        // atomics.  These are all the methods used by GB_accum_mask.  They
        // could also skip constructing the S matrix (Sp == Cp, Sh == Ch, Si ==
        // Ci except for zombies and changing zombies, and an implied Sx [p] ==
        // p even with zombies present).

        //  M   -   -   -   A   ?       06x: C(:,:)<M> = A
        //  M   -   -   +   A   ?       08x: C(:,:)<M> += A
        //  M   -   r   -   A   ?       10x: C(:,:)<M,repl> = A
        //  M   -   r   +   A   ?       12x: C(:,:)<M,repl> += A
        //  M   c   -   -   A   ?       14x: C(:,:)<!M> = A
        //  M   c   -   +   A   ?       16x: C(:,:)<!M> += A
        //  M   c   r   -   A   ?       18x: C(:,:)<!M,repl> = A
        //  M   c   r   +   A   ?       20x: C(:,:)<!M,repl> += A

        //----------------------------------------------------------------------
        // FUTURE::: C<C,s> += x   C==M, update all values, C_replace ignored
        // FUTURE::: C<C,s> = A    C==M, A dense, C_replace ignored
        //----------------------------------------------------------------------

    // For the single case C(I,J)<M>=A, two methods can be used: 06n and 06s.

    #define Istring ((Ikind == GB_ALL) ? ":" : "I")
    #define Jstring ((Jkind == GB_ALL) ? ":" : "J")

    switch (subassign_method)
    {

        //----------------------------------------------------------------------
        // matrix or scalar subassign via GB_bitmap_assign
        //----------------------------------------------------------------------

        case GB_SUBASSIGN_METHOD_BITMAP : 
        {
            // C is bitmap, or is converted to bitmap.  M and A can have any
            // sparsity (if present).
            GBURBLE ("Method: bitmap_subassign ") ;
            GB_OK (GB_bitmap_assign (C, C_replace,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_comp, Mask_struct, accum, A, scalar, scalar_type,
                GB_SUBASSIGN, Werk)) ;
        }
        break ;

        //----------------------------------------------------------------------
        // C = x where x is a scalar; C becomes full
        //----------------------------------------------------------------------

        case GB_SUBASSIGN_METHOD_21 : 
        {

            //  =====================       ==============
            //  M   cmp rpl acc A   S       method: action
            //  =====================       ==============

            //  -   -   x   -   -   -       21:  C = x, no S, C anything

            // Method 21: C = x where x is a scalar; C becomes full
            GBURBLE ("Method 21: (C full) = scalar ") ;
            ASSERT (C->iso) ;
            GB_convert_any_to_full (C) ;
        }
        break ;

        //----------------------------------------------------------------------
        // C = A
        //----------------------------------------------------------------------

        case GB_SUBASSIGN_METHOD_24 : 
        {

            //  =====================       ==============
            //  M   cmp rpl acc A   S       method: action
            //  =====================       ==============

            //  -   -   x   -   A   -       24:  C = A, no S, C and A anything

            // Method 24: C = A
            GBURBLE ("Method 24: C = Z ") ;
            GB_OK (GB_subassign_24 (C, A, Werk)) ;
        }
        break ;

        //----------------------------------------------------------------------
        // C += A or x where C is dense or full (and becomes full)
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   +   -   -       22:  C += x, no S, C dense
        //  -   -   -   +   A   -       23:  C += A, no S, C dense

        case GB_SUBASSIGN_METHOD_22 : 
        {
            // Method 22: C(:,:) += x where C is dense or full
            GBURBLE ("Method 22: (C full) += scalar ") ;
            GB_OK (GB_subassign_22 (C, scalar, scalar_type, accum, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_23 : 
        {
            // Method 23: C(:,:) += A where C is dense or full
            GBURBLE ("Method 23: (C full) += Z ") ;
            GB_OK (GB_subassign_23 (C, A, accum, Werk)) ;
        }
        break ;

        //----------------------------------------------------------------------
        // C(I,J)<M> = scalar or +=scalar
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   -   -   -   -       05d: C(:,:)<M> = x, no S, C dense
        //  M   -   -   -   -   -       05e: C(:,:)<M,s> = x, no S, C empty
        //  M   -   -   -   -   -       05f: C(:,:)<C,s> = x, no S, C == M
        //  M   -   -   -   -   -       05:  C(I,J)<M> = x, no S
        //  M   -   -   +   -   -       07:  C(I,J)<M> += x, no S

        case GB_SUBASSIGN_METHOD_05f : 
        {
            // Method 05f: C(:,:)<C,s> = scalar ; no S; C == M, M structural
            GBURBLE ("Method 05f: C<C,struct> = scalar ") ;
            // no more work to do; all work has been done by GB_assign_prep
        }
        break ;

        case GB_SUBASSIGN_METHOD_05e : 
        {
            // Method 05e: C(:,:)<M> = scalar ; no S; C empty, M structural
            GBURBLE ("Method 05e: (C empty)<M,struct> = scalar ") ;
            GB_OK (GB_subassign_05e (C, M, scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_05d : 
        {
            // Method 05d: C(:,:)<M> = scalar ; no S; C is dense or full;
            // C becomes full.
            GBURBLE ("Method 05d: (C full)<M> = scalar ") ;
            GB_OK (GB_subassign_05d (C,
                M, Mask_struct, scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_05 : 
        {
            // Method 05: C(I,J)<M> = scalar ; no S
            GBURBLE ("Method 05: C(%s,%s)<M> = scalar ; no S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_05 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_07 : 
        {
            // Method 07: C(I,J)<M> += scalar ; no S
            GBURBLE ("Method 07: C(%s,%s)<M> += scalar ; no S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_07 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, accum, scalar, scalar_type, Werk)) ;
        }
        break ;

        //----------------------------------------------------------------------
        // C(I,J)<M> = A or += A
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  C   -   -   +   A   -       27:  C<C,struct> += A
        //  M   -   -   +   A   -       08n: C(I,J)<M> += A, no S
        //  M   -   -   +   A   S       08s: C(I,J)<M> += A, with S
        //  A   -   -   -   A   -       06d: C<A> = A, no S, C dense
        //  M   -   x   -   A   -       25:  C<M,s> = A, A dense, C empty
        //  M   -   -   -   A   -       06n: C(I,J)<M> = A, no S
        //  M   -   -   -   A   S       06s: C(I,J)<M> = A, with S

        case GB_SUBASSIGN_METHOD_27 : 
        {
            // Method 27: C<C,struct> += A ; no S
            GBURBLE ("Method 27: C<C,struct> += Z ; no S ") ;
            GB_OK (GB_subassign_27 (C, accum, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_08n : 
        {
            // Method 08n: C(I,J)<M> += A ; no S
            GBURBLE ("Method 08n: C(%s,%s)<M> += Z ; no S ", Istring, Jstring) ;
            GB_OK (GB_subassign_08n (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, accum, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_08s : 
        {
            // Method 08s: C(I,J)<M> += A ; with S
            GBURBLE ("Method 08s: C(%s,%s)<M> += Z ; with S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_08s_and_16 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, false, Mask_struct, accum, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_06d : 
        {
            // Method 06d: C(:,:)<A> = A ; no S, C dense or full;
            GBURBLE ("Method 06d: (C full)<Z> = Z ") ;
            GB_OK (GB_subassign_06d (C, A, Mask_struct, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_25 : 
        {
            // Method 25:  C<M,struct> = A, A dense, C empty
            // A is dense or full; remains unchanged
            // C is iso if A is so
            GB_BURBLE_DENSE (A, "Method 25: (C empty)<M> = (Z %s) ") ;
            GB_OK (GB_subassign_25 (C, M, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_06n : 
        {
            // Method 06n: C(I,J)<M> = A ; no S
            GBURBLE ("Method 06n: C(%s,%s)<M> = Z ; no S ", Istring, Jstring) ;
            GB_OK (GB_subassign_06n (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_06s : 
        {
            // Method 06s: C(I,J)<M> = A ; using S
            GBURBLE ("Method 06s: C(%s,%s)<M> = Z ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_06s_and_14 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, false, Mask_struct, A, Werk)) ;
        }
        break ;

        //----------------------------------------------------------------------
        // assignment using S_Extraction method (except method 26), no mask M
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   -   -   S       01:  C(I,J) = x, with S
        //  -   -   -   -   A   S       02:  C(I,J) = A, with S
        //  -   -   -   -   A   -       26:  C(:,j1:j2) = A, append cols, no S
        //  -   -   -   +   -   S       03:  C(I,J) += x, with S
        //  -   -   -   +   A   S       04:  C(I,J) += A, with S

        case GB_SUBASSIGN_METHOD_01 : 
        {
            // Method 01: C(I,J) = scalar ; using S
            GBURBLE ("Method 01: C(%s,%s) = scalar ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_01 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_03 : 
        {
            // Method 03: C(I,J) += scalar ; using S
            GBURBLE ("Method 03: C(%s,%s) += scalar ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_03 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                accum, scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_26 : 
        {
            // Method 26: C(:,j1:j2) = A ; append column(s), no S.
            GBURBLE ("Method 26: C(:,j1:j2) = Z ; append") ;
            ASSERT (Ikind == GB_ALL) ;
            ASSERT (Jkind == GB_RANGE) ;
            ASSERT (nJ >= 1) ;
            GB_OK (GB_subassign_26 (C, Jcolon, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_02 : 
        {
            // Method 02: C(I,J) = A ; using S
            GBURBLE ("Method 02: C(%s,%s) = Z ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_02 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_04 : 
        {
            // Method 04: C(I,J) += A ; using S
            GBURBLE ("Method 04: C(%s,%s) += Z ; using S ", Istring, Jstring) ;
            GB_OK (GB_subassign_04 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                accum, A, Werk)) ;
        }
        break ;

        //----------------------------------------------------------------------
        // C(I,J)<#M> = scalar or += scalar ; using S
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   r   -   -   S       09:  C(I,J)<M,repl> = x, with S
        //  M   -   r   +   -   S       11:  C(I,J)<M,repl> += x, with S
        //  M   c   -   -   -   S       13:  C(I,J)<!M> = x, with S
        //  M   c   -   +   -   S       15:  C(I,J)<!M> += x, with S
        //  M   c   r   -   -   S       17:  C(I,J)<!M,repl> = x, with S
        //  M   c   r   +   -   S       19:  C(I,J)<!M,repl> += x, with S

        case GB_SUBASSIGN_METHOD_17 : 
        {
            // Method 17: C(I,J)<!M,repl> = scalar ; using S
            GBURBLE ("Method 17: C(%s,%s)<!M,repl> = scalar ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_17 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_13 : 
        {
            // Method 13: C(I,J)<!M> = scalar ; using S
            GBURBLE ("Method 13: C(%s,%s)<!M> = scalar ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_13 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_09 : 
        {
            // Method 09: C(I,J)<M,repl> = scalar ; using S
            GBURBLE ("Method 09: C(%s,%s)<M,repl> = scalar ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_09 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_19 : 
        {
            // Method 19: C(I,J)<!M,repl> += scalar ; using S
            GBURBLE ("Method 19: C(%s,%s)<!M,repl> += scalar ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_19 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, accum, scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_15 : 
        {
            // Method 15: C(I,J)<!M> += scalar ; using S
            GBURBLE ("Method 15: C(%s,%s)<!M> += scalar ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_15 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, accum, scalar, scalar_type, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_11 : 
        {
            // Method 11: C(I,J)<M,repl> += scalar ; using S
            GBURBLE ("Method 11: C(%s,%s)<M,repl> += scalar ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_11 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, Mask_struct, accum, scalar, scalar_type, Werk)) ;
        }
        break ;

        //------------------------------------------------------------------
        // C(I,J)<#M> = A or += A ; using S
        //------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   r   -   A   S       10:  C(I,J)<M,repl> = A, with S
        //  M   -   r   +   A   S       12:  C(I,J)<M,repl> += A, with S
        //  M   c   -   -   A   S       14:  C(I,J)<!M> = A, with S
        //  M   c   -   +   A   S       16:  C(I,J)<!M> += A, with S
        //  M   c   r   -   A   S       18:  C(I,J)<!M,repl> = A, with S
        //  M   c   r   +   A   S       20:  C(I,J)<!M,repl> += A, with S

        case GB_SUBASSIGN_METHOD_18 : 
        {
            // Method 18: C(I,J)<!M,repl> = A ; using S
            GBURBLE ("Method 18: C(%s,%s)<!M,repl> = Z ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_10_and_18 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, true, Mask_struct, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_14 : 
        {
            // Method 14: C(I,J)<!M> = A ; using S
            GBURBLE ("Method 14: C(%s,%s)<!M> = Z ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_06s_and_14 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, true, Mask_struct, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_10 : 
        {
            // Method 10: C(I,J)<M,repl> = A ; using S
            GBURBLE ("Method 10: C(%s,%s)<M,repl> = Z ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_10_and_18 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, false, Mask_struct, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_20 : 
        {
            // Method 20: C(I,J)<!M,repl> += A ; using S
            GBURBLE ("Method 20: C(%s,%s)<!M,repl> += Z ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_12_and_20 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, true, Mask_struct, accum, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_16 : 
        {
            // Method 16: C(I,J)<!M> += A ; using S
            GBURBLE ("Method 16: C(%s,%s)<!M> += Z ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_08s_and_16 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, true, Mask_struct, accum, A, Werk)) ;
        }
        break ;

        case GB_SUBASSIGN_METHOD_12 : 
        {
            // Method 12: C(I,J)<M,repl> += A ; using S
            GBURBLE ("Method 12: C(%s,%s)<M,repl> += Z ; using S ",
                Istring, Jstring) ;
            GB_OK (GB_subassign_12_and_20 (C,
                I, I_is_32, ni, nI, Ikind, Icolon,
                J, J_is_32, nj, nJ, Jkind, Jcolon,
                M, false, Mask_struct, accum, A, Werk)) ;
        }
        break ;

        default:
            ASSERT (GB_DEAD_CODE) ;
    }

    //--------------------------------------------------------------------------
    // finalize C and return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C subassigner result", GB0) ;
    return (GB_block (C, Werk)) ;
}

