//------------------------------------------------------------------------------
// GB_subassigner_method: determine method for GB_subassign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_subassign.h"

int GB_subassigner_method           // return method to use in GB_subassigner
(
    // outputs
    bool *C_iso_out,                // true if C is iso on output
    GB_void *cout,                  // iso value of C on output
    // inputs
    const GrB_Matrix C,             // input/output matrix for results
    const bool C_replace,           // C matrix descriptor
    const GrB_Matrix M,             // optional mask for C(I,J), unused if NULL
    const bool Mask_comp,           // mask descriptor
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),A)
    const GrB_Matrix A,             // input matrix (NULL for scalar expansion)
    const int Ikind,
    const int Jkind,
    const bool scalar_expansion,    // if true, expand scalar to A
    const void *scalar,
    const GrB_Type scalar_type      // type of the scalar, or NULL
)
{

    //--------------------------------------------------------------------------
    // get all properties of C, M, and A required by this function
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    // empty_mask: mask not present and complemented.  This condition has
    // already handled by GB_assign_prep.
    bool empty_mask = (M == NULL) && Mask_comp ;
    ASSERT (!empty_mask) ;
    #endif

    // no_mask: mask not present and not complemented
    bool no_mask = (M == NULL) && !Mask_comp ;
    bool M_is_A = GB_all_aliased (M, A) ;
    bool M_is_bitmap = GB_IS_BITMAP (M) ;

    bool A_is_bitmap = GB_IS_BITMAP (A) ;
    bool A_is_full = GB_IS_FULL (A) ;
    int64_t anz = GB_nnz (A) ;

    // these properties of C are not affected by wait(C):
    bool C_is_M = (C == M) ;
    GrB_Type ctype = C->type ;

    // these properties of C can change after wait(C):
    bool C_is_empty = (GB_nnz (C) == 0 && !GB_PENDING (C) && !GB_ZOMBIES (C)) ;
    bool C_is_bitmap = GB_IS_BITMAP (C) ;
    bool C_is_full = GB_IS_FULL (C) ;

    //--------------------------------------------------------------------------
    // determine the method to use
    //--------------------------------------------------------------------------

    // whole_C_matrix is true if all of C(:,:) is being assigned to
    bool whole_C_matrix = (Ikind == GB_ALL) && (Jkind == GB_ALL) ;

    bool C_splat_scalar = false ;   // C(:,:) = x
    bool C_splat_matrix = false ;   // C(:,:) = A

    if (whole_C_matrix && no_mask && (accum == NULL))
    {
        // C(:,:) = x or A:  whole matrix assignment with no mask
        if (scalar_expansion)
        { 
            // Method 21: C(:,:) = x
            C_splat_scalar = true ;
        }
        else
        { 
            // Method 24: C(:,:) = A
            C_splat_matrix = true ;
        }
    }

    // check if C is full
    bool C_dense_update = false ;
    if (C_is_full && whole_C_matrix && no_mask && (accum != NULL)
            && (ctype == accum->ztype) && (ctype == accum->xtype))
    { 
        // C(:,:) += x or A, where C is full, no typecasting of C
        C_dense_update = true ;
    }

    // GB_assign_prep has already disabled C_replace if no mask present
    ASSERT (GB_IMPLIES (no_mask, !C_replace)) ;

    // if C is empty, C_replace is effectively false and already disabled
    ASSERT (GB_IMPLIES (C_is_empty, !C_replace)) ;

    // simple_mask: C(I,J)<M> = ... ; or C(I,J)<M> += ...
    bool simple_mask = (!C_replace && M != NULL && !Mask_comp) ;

    // C_Mask_scalar: C(I,J)<M> = scalar or += scalar
    bool C_Mask_scalar = (scalar_expansion && simple_mask) ;

    // C_Mask_matrix:  C(I,J)<M> = A or += A
    bool C_Mask_matrix = (!scalar_expansion && simple_mask) ;

    bool S_Extraction ;
    bool method_06d = false ;
    bool method_25 = false ;

    if (C_splat_scalar)
    { 
        // Method 21: C(:,:) = x where x is a scalar; C becomes full
        S_Extraction = false ;
    }
    else if (C_splat_matrix)
    { 
        // Method 24: C(:,:) = A
        S_Extraction = false ;
    }
    else if (C_dense_update)
    { 
        // Methods 22 and 23: C(:,:) += x or A where C is full
        S_Extraction = false ;
    }
    else if (C_Mask_scalar)
    { 
        // Method 05*, or 07: C(I,J)<M> = or += scalar; C_replace false
        S_Extraction = false ;
    }
    else if (C_Mask_matrix)
    {
        // C(I,J)<M> = A or += A
        if (accum != NULL)
        { 
            // Method 08n: C(I,J)<M> += A, no S.  Cannot use M or A as bitmap.
            // Method 08s: C(I,J)<M> += A, with S.  Can use M or A as bitmap.
            // if S_Extraction is true, Method 08s is used (with S).
            // Method 08n is not used if any matrix is bitmap.
            // If C is bitmap, GB_bitmap_assign_M_accum is used instead.
            S_Extraction = M_is_bitmap || A_is_bitmap ;
        }
        else
        {
            // C(I,J)<M> = A ;  use 06s (with S) or 06n (without S)
            // method 06s (with S) is faster when nnz (A) < nnz (M).
            if ((C_is_full || C_is_bitmap) && whole_C_matrix && M_is_A)
            { 
                // Method 06d: C<A> = A
                method_06d = true ;
                S_Extraction = false ;
            }
            else if (C_is_empty && whole_C_matrix && Mask_struct &&
                (A_is_full || A_is_bitmap))
            { 
                // Method 25: C<M,s> = A, where M is structural, A is
                // full, and C starts out empty.  The pattern of C will be the
                // same as M, and the subassign method is extremely simple.
                method_25 = true ;
                S_Extraction = false ;
            }
            else
            { 
                // Method 06n (no S) or Method 06s (with S):
                // Method 06n is not used if M or A are bitmap.  If M and A are
                // aliased and Method 06d is not used, then 06s is used instead
                // of 06n since M==A implies nnz(A) == nnz(M).
                S_Extraction = anz < GB_nnz (M) || M_is_bitmap || A_is_bitmap ;
            }
        }
    }
    else
    { 
        // all other methods require S
        S_Extraction = true ;
    }

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
    // with additional special cases when C(:,:) is full.

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
        //  -   -   -   +   -   -       22:  C += x, no S, C full
        //  -   -   -   +   A   -       23:  C += A, no S, C full

        //  -   -   -   -   -   S       01:  C(I,J) = x, with S
        //  -   -   -   -   A   S       02:  C(I,J) = A, with S
        //  -   -   -   +   -   S       03:  C(I,J) += x, with S
        //  -   -   -   +   A   S       04:  C(I,J) += A, with S
        //  -   -   r                        uses methods 01, 02, 03, 04
        //  -   c   -                        no work to do
        //  -   c   r           S       00:  C(I,J)<!,repl> = empty, with S

        //  M   -   -   -   -   -       05d: C<M> = x, no S, C full
        //  M   -   -   -   -   -       05e: C<M,s> = x, no S, C empty
        //  M   -   -   -   -   -       05f: C<C,s> = x, no S, C == M
        //  M   -   -   -   -   -       05:  C(I,J)<M> = x, no S
        //  A   -   -   -   A   -       06d: C<A> = A, no S, C full
        //  M   -   -   -   A   -       25:  C<M,s> = A, A full, C empty
        //  M   -   -   -   A   -       06n: C(I,J)<M> = A, no S
        //  M   -   -   -   A   S       06s: C(I,J)<M> = A, with S
        //  M   -   -   +   -   -       07:  C(I,J)<M> += x, no S
        //  M   -   -   +   A   -       08n: C(I,J)<M> += A, no S
        //  M   -   -   +   A   -       08s: C(I,J)<M> += A, with S
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

        //  M   -   -   -   A   -       06x: C(:,:)<M> = A
        //  M   -   -   +   A   -       08x: C(:,:)<M> += A
        //  M   -   r   -   A   -       10x: C(:,:)<M,repl> = A
        //  M   -   r   +   A   -       12x: C(:,:)<M,repl> += A
        //  M   c   -   -   A   -       14x: C(:,:)<!M> = A
        //  M   c   -   +   A   -       16x: C(:,:)<!M> += A
        //  M   c   r   -   A   -       18x: C(:,:)<!M,repl> = A
        //  M   c   r   +   A   -       20x: C(:,:)<!M,repl> += A

        //----------------------------------------------------------------------
        // FUTURE::: C<C,s> += x   C == M, update all values, C_replace ignored
        // FUTURE::: C<C,s> = A    C == M, A full, C_replace ignored
        //----------------------------------------------------------------------

    // For the single case C(I,J)<M>=A, two methods can be used: 06n and 06s.

    int subassign_method = -1 ;

    if (C_splat_scalar)
    { 

        //----------------------------------------------------------------------
        // C = x where x is a scalar; C becomes full
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============

        //  -   -   x   -   -   -       21:  C = x, no S, C anything

        ASSERT (whole_C_matrix) ;           // C(:,:) is modified
        ASSERT (M == NULL) ;                // no mask present
        ASSERT (accum == NULL) ;            // accum is not present
        ASSERT (!C_replace) ;               // C_replace is effectively false
        ASSERT (!S_Extraction) ;            // S is not used
        ASSERT (scalar_expansion) ;         // x is a scalar

        // Method 21: C = x where x is a scalar; C becomes full
        subassign_method = GB_SUBASSIGN_METHOD_21 ;

    }
    else if (C_splat_matrix)
    { 

        //----------------------------------------------------------------------
        // C = A
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============

        //  -   -   x   -   A   -       24:  C = A, no S, C and A anything

        ASSERT (whole_C_matrix) ;           // C(:,:) is modified
        ASSERT (M == NULL) ;                // no mask present
        ASSERT (accum == NULL) ;            // accum is not present
        ASSERT (!C_replace) ;               // C_replace is effectively false
        ASSERT (!S_Extraction) ;            // S is not used
        ASSERT (!scalar_expansion) ;        // A is a matrix

        // Method 24: C = A
        subassign_method = GB_SUBASSIGN_METHOD_24 ;

    }
    else if (C_dense_update)
    {

        //----------------------------------------------------------------------
        // C += A or x where C is full
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   +   -   -       22:  C += x, no S, C full
        //  -   -   -   +   A   -       23:  C += A, no S, C full

        ASSERT (C_is_full) ;                // C is full
        ASSERT (whole_C_matrix) ;           // C(:,:) is modified
        ASSERT (M == NULL) ;                // no mask present
        ASSERT (accum != NULL) ;            // accum is present
        ASSERT (!C_replace) ;               // C_replace is false
        ASSERT (!S_Extraction) ;            // S is not used

        if (scalar_expansion)
        { 
            // Method 22: C(:,:) += x where C is full
            subassign_method = GB_SUBASSIGN_METHOD_22 ;
        }
        else
        { 
            // Method 23: C(:,:) += A where C is full
            subassign_method = GB_SUBASSIGN_METHOD_23 ;
        }

    }
    else if (C_Mask_scalar)
    {

        //----------------------------------------------------------------------
        // C(I,J)<M> = scalar or +=scalar
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   -   -   -   -       05d: C(:,:)<M> = x, no S, C full
        //  M   -   -   -   -   -       05e: C(:,:)<M,s> = x, no S, C empty
        //  M   -   -   -   -   -       05f: C(:,:)<C,s> = x, no S, C == M
        //  M   -   -   -   -   -       05:  C(I,J)<M> = x, no S
        //  M   -   -   +   -   -       07:  C(I,J)<M> += x, no S

        ASSERT (scalar_expansion) ;         // A is a scalar
        ASSERT (M != NULL && !Mask_comp) ;  // mask M present, not compl.
        ASSERT (!C_replace) ;               // C_replace is false
        ASSERT (!S_Extraction) ;            // S is not used

        if (accum == NULL)
        {
            if (C_is_M && whole_C_matrix && Mask_struct)
            { 
                // Method 05f: C(:,:)<C,s> = scalar ; no S ; C == M ; M struct
                subassign_method = GB_SUBASSIGN_METHOD_05f ;
            }
            else if (C_is_empty && whole_C_matrix && Mask_struct)
            { 
                // Method 05e: C(:,:)<M,s> = scalar ; no S; C empty, M struct
                subassign_method = GB_SUBASSIGN_METHOD_05e ;
            }
            else if (C_is_full && whole_C_matrix)
            { 
                // Method 05d: C(:,:)<M> = scalar ; no S; C is full
                // C becomes full.
                subassign_method = GB_SUBASSIGN_METHOD_05d ;
            }
            else
            { 
                // Method 05: C(I,J)<M> = scalar ; no S
                subassign_method = GB_SUBASSIGN_METHOD_05 ;
            }
        }
        else
        { 
            // Method 07: C(I,J)<M> += scalar ; no S
            subassign_method = GB_SUBASSIGN_METHOD_07 ;
        }

    }
    else if (C_Mask_matrix)
    {

        //----------------------------------------------------------------------
        // C(I,J)<M> = A or += A
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  M   -   -   +   A   -       08n:  C(I,J)<M> += A, no S
        //  M   -   -   +   A   -       08s:  C(I,J)<M> += A, with S
        //  A   -   -   -   A   -       06d: C<A> = A, no S, C full
        //  M   -   x   -   A   -       25:  C<M,s> = A, A full, C empty
        //  M   -   -   -   A   -       06n: C(I,J)<M> = A, no S
        //  M   -   -   -   A   S       06s: C(I,J)<M> = A, with S

        ASSERT (!scalar_expansion) ;        // A is a matrix
        ASSERT (M != NULL && !Mask_comp) ;  // mask M present, not compl.
        ASSERT (!C_replace) ;

        if (accum != NULL)
        {
            if (S_Extraction)
            { 
                // Method 08s: C(I,J)<M> += A ; with S
                subassign_method = GB_SUBASSIGN_METHOD_08s ;
            }
            else
            { 
                // Method 08n: C(I,J)<M> += A ; no S
                // No matrix can be bitmap.
                subassign_method = GB_SUBASSIGN_METHOD_08n ;
            }
        }
        else if (method_06d)
        { 
            // Method 06d: C(:,:)<A> = A ; no S, C full
            subassign_method = GB_SUBASSIGN_METHOD_06d ;
            ASSERT ((C_is_full || C_is_bitmap) && whole_C_matrix && M_is_A) ;
        }
        else if (method_25)
        { 
            // Method 25: C<M,struct> = A, C empty; A is full or bitmap
            subassign_method = GB_SUBASSIGN_METHOD_25 ;
        }
        else if (!S_Extraction)
        { 
            // Method 06n: C(I,J)<M> = A ; no S
            // If M or A are bitmap, this method is not used;
            // 06s is used instead.
            subassign_method = GB_SUBASSIGN_METHOD_06n ;
        }
        else
        { 
            // Method 06s: C(I,J)<M> = A ; using S
            subassign_method = GB_SUBASSIGN_METHOD_06s ;
        }

    }
    else if (M == NULL)
    {

        //----------------------------------------------------------------------
        // assignment using S_Extraction method, no mask M
        //----------------------------------------------------------------------

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============
        //  -   -   -   -   -   S       01:  C(I,J) = x, with S
        //  -   -   -   -   A   S       02:  C(I,J) = A, with S
        //  -   -   -   +   -   S       03:  C(I,J) += x, with S
        //  -   -   -   +   A   S       04:  C(I,J) += A, with S

        ASSERT (!Mask_comp) ;
        ASSERT (!C_replace) ;
        ASSERT (S_Extraction) ;            // S is used

        if (scalar_expansion)
        {
            if (accum == NULL)
            { 
                // Method 01: C(I,J) = scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_01 ;
            }
            else
            { 
                // Method 03: C(I,J) += scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_03 ;
            }
        }
        else
        {
            if (accum == NULL)
            { 
                // Method 02: C(I,J) = A ; using S
                subassign_method = GB_SUBASSIGN_METHOD_02 ;
            }
            else
            { 
                // Method 04: C(I,J) += A ; using S
                subassign_method = GB_SUBASSIGN_METHOD_04 ;
            }
        }

    }
    else if (scalar_expansion)
    {

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

        ASSERT (!C_Mask_scalar) ;
        ASSERT (C_replace || Mask_comp) ;
        ASSERT (S_Extraction) ;            // S is used

        if (accum == NULL)
        {
            if (Mask_comp && C_replace)
            { 
                // Method 17: C(I,J)<!M,repl> = scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_17 ;
            }
            else if (Mask_comp)
            { 
                // Method 13: C(I,J)<!M> = scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_13 ;
            }
            else // if (C_replace)
            { 
                // Method 09: C(I,J)<M,repl> = scalar ; using S
                ASSERT (C_replace) ;
                subassign_method = GB_SUBASSIGN_METHOD_09 ;
            }
        }
        else
        {
            if (Mask_comp && C_replace)
            { 
                // Method 19: C(I,J)<!M,repl> += scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_19 ;
            }
            else if (Mask_comp)
            { 
                // Method 15: C(I,J)<!M> += scalar ; using S
                subassign_method = GB_SUBASSIGN_METHOD_15 ;
            }
            else // if (C_replace)
            { 
                // Method 11: C(I,J)<M,repl> += scalar ; using S
                ASSERT (C_replace) ;
                subassign_method = GB_SUBASSIGN_METHOD_11 ;
            }
        }

    }
    else
    {

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

        ASSERT (Mask_comp || C_replace) ;
        ASSERT (S_Extraction) ;            // S is used

        if (accum == NULL)
        {
            if (Mask_comp && C_replace)
            { 
                // Method 18: C(I,J)<!M,repl> = A ; using S
                subassign_method = GB_SUBASSIGN_METHOD_18 ;
            }
            else if (Mask_comp)
            { 
                // Method 14: C(I,J)<!M> = A ; using S
                subassign_method = GB_SUBASSIGN_METHOD_14 ;
            }
            else // if (C_replace)
            { 
                // Method 10: C(I,J)<M,repl> = A ; using S
                ASSERT (C_replace) ;
                subassign_method = GB_SUBASSIGN_METHOD_10 ;
            }
        }
        else
        {
            if (Mask_comp && C_replace)
            { 
                // Method 20: C(I,J)<!M,repl> += A ; using S
                subassign_method = GB_SUBASSIGN_METHOD_20 ;
            }
            else if (Mask_comp)
            { 
                subassign_method = GB_SUBASSIGN_METHOD_16 ;
            }
            else // if (C_replace)
            { 
                // Method 12: C(I,J)<M,repl> += A ; using S
                ASSERT (C_replace) ;
                subassign_method = GB_SUBASSIGN_METHOD_12 ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // determine the iso property of C on output
    //--------------------------------------------------------------------------

    // For scalar expansion, or if A is iso on input, then C might be iso on
    // output.  Otherwise, C is always non-iso on output.  Skip this if cout or
    // C_iso_out are NULL, since that means they have already been computed.

    bool iso_check = (cout != NULL && C_iso_out != NULL) ;
    if (iso_check)
    {

        bool A_iso = scalar_expansion           // all scalars are iso
            || (A != NULL && A->iso)            // or A is iso
            || (anz == 1 && !A_is_bitmap) ;     // or A is effectively iso
        if (A_iso)
        {

            //------------------------------------------------------------------
            // cout = tentative iso value of C on output
            //------------------------------------------------------------------

            GB_Type_code ccode = ctype->code ;
            size_t       csize = ctype->size ;
            GrB_Type atype = (A == NULL) ? scalar_type : A->type ;
            GB_Type_code acode = atype->code ;
            size_t       asize = atype->size ;

            // cout = (ctype) (scalar or A->x)
            GB_cast_scalar (cout, ccode, (scalar_expansion) ? scalar : A->x,
                acode, asize) ;
            bool c_ok = false ;
            if (C_is_empty)
            { 
                // C is empty on input; note that C->iso might also be true,
                // but this is ignored.
                c_ok = true ;
            }
            else if (C->iso)
            { 
                // C is iso on input; compare cout and C->x
                c_ok = (memcmp (cout, C->x, csize) == 0) ;
            }

            //------------------------------------------------------------------
            // apply the accum, if present, and compare its result with cout
            //------------------------------------------------------------------

            bool accum_ok = false ;
            if (c_ok && accum != NULL && C->iso)
            {
                if (C_is_empty)
                { 
                    // If C is empty, the accum is not applied.
                    accum_ok = true ;
                }
                else
                { 
                    // C is iso and not empty; check the result of accum
                    GxB_binary_function faccum = accum->binop_function ;

                    size_t xsize = accum->xtype->size ;
                    size_t ysize = accum->ytype->size ;
                    size_t zsize = accum->ztype->size ;

                    GB_Type_code xcode = accum->xtype->code ;
                    GB_Type_code ycode = accum->ytype->code ;
                    GB_Type_code zcode = accum->ztype->code ;

                    // x = (xtype) C->x
                    GB_void x [GB_VLA(xsize)] ;
                    GB_cast_scalar (x, xcode, C->x, ccode, csize) ;

                    // y = (ytype) (scalar or A->x)
                    GB_void y [GB_VLA(ysize)] ;
                    GB_cast_scalar (y, ycode,
                        (scalar_expansion) ? scalar : A->x, acode, asize) ;

                    // z = x + y
                    GB_void z [GB_VLA(zsize)] ;
                    faccum (z, x, y) ;

                    // c = (ctype) z
                    GB_void c [GB_VLA(csize)] ;
                    GB_cast_scalar (c, ccode, z, zcode, zsize) ;

                    // compare c and cout
                    accum_ok = (memcmp (cout, c, csize) == 0) ;
                }
            }

            switch (subassign_method)
            {

                //--------------------------------------------------------------
                // C_out is iso if C_in empty, or C_in iso and cin == scalar
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_01 :   // C(I,J) = scalar
                case GB_SUBASSIGN_METHOD_05 :   // C(I,J)<M> = scalar
                case GB_SUBASSIGN_METHOD_13 :   // C(I,J)<!M> = scalar
                case GB_SUBASSIGN_METHOD_05d :  // C(:,:)<M> = scalar ; C full
                case GB_SUBASSIGN_METHOD_09 :   // C(I,J)<M,replace> = scalar
                case GB_SUBASSIGN_METHOD_17 :   // C(I,J)<!M,replace> = scalar
                    (*C_iso_out) = c_ok ;
                    break ;

                //--------------------------------------------------------------
                // C_out is iso if C_in empty, or C_in iso and cin == a
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_02 :   // C(I,J) = A
                case GB_SUBASSIGN_METHOD_06s :  // C(I,J)<M> = A ; with S
                case GB_SUBASSIGN_METHOD_14 :   // C(I,J)<!M> = A
                case GB_SUBASSIGN_METHOD_10 :   // C(I,J)<M,replace> = A
                case GB_SUBASSIGN_METHOD_18 :   // C(I,J)<!M,replace> = A
                case GB_SUBASSIGN_METHOD_06d :  // C(:,:)<A> = A ; C is full
                case GB_SUBASSIGN_METHOD_06n :  // C(I,J)<M> = A ; no S
                    (*C_iso_out) = c_ok ;
                    break ;

                //--------------------------------------------------------------
                // C_out is always iso, regardless of C_in
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_21 :   // C(:,:) = scalar
                case GB_SUBASSIGN_METHOD_05e :  // C(:,:)<M,struct>=x ; C empty
                case GB_SUBASSIGN_METHOD_05f :  // C(:,:)<C,struct>=scalar
                    (*C_iso_out) = true ;       // scalars are always iso
                    break ;

                //--------------------------------------------------------------
                // C_out is iso if A is iso, regardless of C_in
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_24 :   // C = A
                case GB_SUBASSIGN_METHOD_25 :   // C(:,:)<M,str> = A ; C empty
                    (*C_iso_out) = true ;       // A is iso (see above)
                    break ;

                //--------------------------------------------------------------
                // C_out is iso if C_in empty, or C_in iso and cin == cin+scalar
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_03 :   // C(I,J) += scalar
                case GB_SUBASSIGN_METHOD_07 :   // C(I,J)<M> += scalar
                case GB_SUBASSIGN_METHOD_15 :   // C(I,J)<!M> += scalar
                case GB_SUBASSIGN_METHOD_22 :   // C += scalar ; C is full
                case GB_SUBASSIGN_METHOD_11 :   // C(I,J)<M,replace> += scalar
                case GB_SUBASSIGN_METHOD_19 :   // C(I,J)<!M,replace> += scalar
                    (*C_iso_out) = accum_ok ;
                    break ;

                //--------------------------------------------------------------
                // C_out is iso if C_in empty, or C_in and A iso and cin==cin+a
                //--------------------------------------------------------------

                case GB_SUBASSIGN_METHOD_12 :   // C(I,J)<M,replace> += A
                case GB_SUBASSIGN_METHOD_20 :   // C(I,J)<!M,replace> += A
                case GB_SUBASSIGN_METHOD_04 :   // C(I,J) += A
                case GB_SUBASSIGN_METHOD_08s :  // C(I,J)<M> += A, with S
                case GB_SUBASSIGN_METHOD_16 :   // C(I,J)<!M> += A 
                case GB_SUBASSIGN_METHOD_23 :   // C += A ; C is full
                case GB_SUBASSIGN_METHOD_08n :  // C(I,J)<M> += A, no S
                    (*C_iso_out) = accum_ok ;
                    break ;

                default :;
            }
        }
        else
        { 
            // A is non-iso, so C is non-iso on output, and cout is not
            // computed
            (*C_iso_out) = false ;
        }
    }

    //--------------------------------------------------------------------------
    // determine if the subassign method can handle this case for bitmaps
    //--------------------------------------------------------------------------

    #define GB_USE_BITMAP_IF(condition) \
        if (condition) subassign_method = GB_SUBASSIGN_METHOD_BITMAP ;

    switch (subassign_method)
    {

        //----------------------------------------------------------------------
        // scalar assignent methods
        //----------------------------------------------------------------------

        case GB_SUBASSIGN_METHOD_01 :   // C(I,J) = scalar
        case GB_SUBASSIGN_METHOD_03 :   // C(I,J) += scalar
        case GB_SUBASSIGN_METHOD_05 :   // C(I,J)<M> = scalar
        case GB_SUBASSIGN_METHOD_07 :   // C(I,J)<M> += scalar
        case GB_SUBASSIGN_METHOD_13 :   // C(I,J)<!M> = scalar
        case GB_SUBASSIGN_METHOD_15 :   // C(I,J)<!M> += scalar
        case GB_SUBASSIGN_METHOD_21 :   // C(:,:) = scalar
            // M can have any sparsity structure, including bitmap
            GB_USE_BITMAP_IF (C_is_bitmap) ;
            break ;

        case GB_SUBASSIGN_METHOD_05d :  // C(:,:)<M> = scalar ; C is full
        case GB_SUBASSIGN_METHOD_05e :  // C(:,:)<M,struct> = scalar ; C empty
        case GB_SUBASSIGN_METHOD_05f :  // C(:,:)<C,struct> = scalar
        case GB_SUBASSIGN_METHOD_22 :   // C += scalar ; C is full
            // C and M can have any sparsity pattern, including bitmap
            break ;

        case GB_SUBASSIGN_METHOD_09 :   // C(I,J)<M,replace> = scalar
        case GB_SUBASSIGN_METHOD_11 :   // C(I,J)<M,replace> += scalar
        case GB_SUBASSIGN_METHOD_17 :   // C(I,J)<!M,replace> = scalar
        case GB_SUBASSIGN_METHOD_19 :   // C(I,J)<!M,replace> += scalar
            // M can have any sparsity structure, including bitmap
            GB_USE_BITMAP_IF (C_is_bitmap || C_is_full) ;
            break ;

        //----------------------------------------------------------------------
        // matrix assignent methods
        //----------------------------------------------------------------------

        // GB_accum_mask may use any of these methods, with I and J as GB_ALL.

        case GB_SUBASSIGN_METHOD_02 :   // C(I,J) = A
        case GB_SUBASSIGN_METHOD_06s :  // C(I,J)<M> = A ; with S
        case GB_SUBASSIGN_METHOD_14 :   // C(I,J)<!M> = A
        case GB_SUBASSIGN_METHOD_10 :   // C(I,J)<M,replace> = A
        case GB_SUBASSIGN_METHOD_18 :   // C(I,J)<!M,replace> = A
        case GB_SUBASSIGN_METHOD_12 :   // C(I,J)<M,replace> += A
        case GB_SUBASSIGN_METHOD_20 :   // C(I,J)<!M,replace> += A
            // M can have any sparsity structure, including bitmap
            GB_USE_BITMAP_IF (C_is_bitmap || C_is_full) ;
            break ;

        case GB_SUBASSIGN_METHOD_04 :   // C(I,J) += A
        case GB_SUBASSIGN_METHOD_08s :  // C(I,J)<M> += A, with S
        case GB_SUBASSIGN_METHOD_16 :   // C(I,J)<!M> += A 
        case GB_SUBASSIGN_METHOD_24 :   // C = A
            // M can have any sparsity structure, including bitmap
            GB_USE_BITMAP_IF (C_is_bitmap) ;
            break ;

        case GB_SUBASSIGN_METHOD_06d :  // C(:,:)<A> = A ; C is full
        case GB_SUBASSIGN_METHOD_23 :   // C += A ; C is full
            // C, M, and A can have any sparsity structure, including bitmap
            break ;

        case GB_SUBASSIGN_METHOD_25 :   // C(:,:)<M,struct> = A ; C empty
            // C, M, and A can have any sparsity structure, including bitmap,
            // but if M is bitmap or full, use bitmap assignment instead.
            GB_USE_BITMAP_IF (M_is_bitmap || GB_IS_FULL (M)) ;
            break ;

        case GB_SUBASSIGN_METHOD_06n :  // C(I,J)<M> = A ; no S
            // If M or A are bitmap, Method 06s is used instead of 06n.
            GB_USE_BITMAP_IF (C_is_bitmap || C_is_full) ;
            ASSERT (!M_is_bitmap) ;
            ASSERT (!A_is_bitmap) ;
            break ;

        case GB_SUBASSIGN_METHOD_08n :  // C(I,J)<M> += A, no S
            // Method 08s is used instead of 08n if M or A are bitmap.
            GB_USE_BITMAP_IF (C_is_bitmap) ;
            ASSERT (!M_is_bitmap) ;
            ASSERT (!A_is_bitmap) ;
            break ;

        // case GB_SUBASSIGN_METHOD_BITMAP:
        default :;
            subassign_method = GB_SUBASSIGN_METHOD_BITMAP ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (subassign_method) ;
}

