//------------------------------------------------------------------------------
// GB_setElement: C(row,col) = scalar or += scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Sets the value of single scalar, C(row,col) = scalar, or C(row,col)+=scalar,
// typecasting from the type of scalar to the type of C, as needed.  Not
// user-callable; does the work for all GrB_*_setElement* functions, and for
// GrB_*assign when a single entry is modified.

// If C(row,col) is already present in the matrix, its value is overwritten
// with the scalar.  Otherwise, if the mode determined by GrB_init is
// non-blocking, the tuple (i,j,scalar) is appended to a list of pending tuples
// to C.  GB_wait assembles these pending tuples.

// GB_setElement when accum is NULL is used by GrB_*_setElement.  It is the
// same as GrB_*assign with an implied SECOND accum operator whose ztype,
// xtype, and ytype are the same as C, with I=i, J=j, a 1-by-1 dense matrix A
// (where nnz (A) == 1), no mask, mask not complemented, C_replace effectively
// false (its value is ignored), and A transpose effectively false (since
// transposing a scalar has no effect).

// GB_setElement when accum is not NULL uses the accum operator instead of the
// implied SECOND operator.  It is used by GrB_*_assign, as a special case.

// Compare this function with GrB_*_extractElement_*

#include "pending/GB_Pending.h"

#define GB_FREE_ALL ;

GrB_Info GB_setElement              // set a single entry, C(row,col) = scalar
(
    GrB_Matrix C,                   // matrix to modify
    const GrB_BinaryOp accum,       // if NULL: C(row,col) = scalar
                                    // else: C(row,col) += scalar
    const void *scalar,             // scalar to set
    const GrB_Index row,            // row index
    const GrB_Index col,            // column index
    const GB_Type_code scalar_code, // type of the scalar
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL) ;
    GB_RETURN_IF_NULL (scalar) ;

    if (row >= GB_NROWS (C))
    { 
        GB_ERROR (GrB_INVALID_INDEX,
            "Row index " GBu " out of range; must be < " GBd,
            row, GB_NROWS (C)) ;
    }
    if (col >= GB_NCOLS (C))
    { 
        GB_ERROR (GrB_INVALID_INDEX,
            "Column index " GBu " out of range; must be < " GBd,
            col, GB_NCOLS (C)) ;
    }

    ASSERT (scalar_code <= GB_UDT_code) ;

    GrB_Type ctype = C->type ;
    GB_Type_code ccode = ctype->code ;

    // scalar_code and C must be compatible
    if (!GB_code_compatible (scalar_code, ccode))
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Input scalar of type [%s]\n"
            "cannot be typecast to entry of type [%s]",
            GB_code_string (scalar_code), ctype->name) ;
    }

    if (accum != NULL)
    { 
        // C and scalar must be compatible with the accum operator
        GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
        GB_OK (GB_BinaryOp_compatible (accum, ctype, ctype, NULL, scalar_code,
            Werk)) ;
    }

    // pending tuples and zombies are expected, and C might be jumbled too
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_PENDING_OK (C)) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;

    //--------------------------------------------------------------------------
    // sort C if needed; do not assemble pending tuples or kill zombies yet
    //--------------------------------------------------------------------------

    if (C->jumbled)
    { 
        GB_OK (GB_wait (C, "C (setElement:jumbled)", Werk)) ;
    }

    // zombies and pending tuples are still OK, but C is no longer jumbled
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (GB_PENDING_OK (C)) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;

    bool C_is_full = GB_IS_FULL (C) ;

    //--------------------------------------------------------------------------
    // check if C needs to convert to non-iso, or if C is a new iso matrix
    //--------------------------------------------------------------------------

    // stype is the type of this scalar
    GrB_Type stype = GB_code_type (scalar_code, ctype) ;
    size_t csize = ctype->size ;

    if (C->iso)
    {

        //----------------------------------------------------------------------
        // typecast the scalar and compare with the iso value of C
        //----------------------------------------------------------------------

        bool convert_to_non_iso ;
        if (accum != NULL)
        { 
            // C(i,j) += scalar always converts C to non-iso
            convert_to_non_iso = true ;
        }
        else if (ctype != stype)
        { 
            // s = (ctype) scalar
            GB_void s [GB_VLA(csize)] ;
            GB_cast_scalar (s, ccode, scalar, scalar_code, csize) ;
            // compare s with the iso value of C
            convert_to_non_iso = (memcmp (C->x, s, csize) != 0) ;
        }
        else
        { 
            // compare the scalar with the iso value of C
            convert_to_non_iso = (memcmp (C->x, scalar, csize) != 0) ;
        }

        if (convert_to_non_iso)
        { 
            // The new entry differs from the iso value of C.  Assemble all
            // pending tuples and convert C to non-iso.  Zombies are OK.
            if (C->Pending != NULL)
            { 
                GB_OK (GB_wait (C, "C (setElement:to non-iso)", Werk)) ;
            }
            GB_OK (GB_convert_any_to_non_iso (C, true)) ;
        }

    }
    else if (GB_nnz (C) == 0 && !C_is_full && C->Pending == NULL
        && accum == NULL)
    {

        //----------------------------------------------------------------------
        // C is empty: this is the first setElement, convert C to iso
        //----------------------------------------------------------------------

        if (ctype != stype)
        { 
            // s = (ctype) scalar
            GB_void s [GB_VLA(csize)] ;
            GB_cast_scalar (s, ccode, scalar, scalar_code, csize) ;
            GB_OK (GB_convert_any_to_iso (C, s)) ;
        }
        else
        { 
            GB_OK (GB_convert_any_to_iso (C, (GB_void *) scalar)) ;
        }
    }

    //--------------------------------------------------------------------------
    // handle the CSR/CSC format
    //--------------------------------------------------------------------------

    int64_t i, j ;
    if (C->is_csc)
    { 
        // set entry with index i in vector j
        i = row ;
        j = col ;
    }
    else
    { 
        // set entry with index j in vector i
        i = col ;
        j = row ;
    }

    int64_t pleft ;
    bool found = false ;
    bool is_zombie ;
    bool C_is_bitmap = GB_IS_BITMAP (C) ;
    C_is_full = GB_IS_FULL (C) ;

    if (C_is_full || C_is_bitmap)
    { 

        //----------------------------------------------------------------------
        // C is bitmap or full
        //----------------------------------------------------------------------

        pleft = i + j * C->vlen ;
        found = true ;
        is_zombie = false ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C is sparse or hypersparse
        //----------------------------------------------------------------------

        int64_t pC_start, pC_end ;
        const int64_t *restrict Ch = C->h ;
        const int64_t *restrict Cp = C->p ;
        if (C->nvals == 0)
        { 
            // C is empty
            found = false ;
        }
        else if (Ch != NULL)
        {
            // C is hypersparse, with at least one entry
            const int64_t *restrict C_Yp = (C->Y == NULL) ? NULL : C->Y->p ;
            const int64_t *restrict C_Yi = (C->Y == NULL) ? NULL : C->Y->i ;
            const int64_t *restrict C_Yx = (C->Y == NULL) ? NULL : C->Y->x ;
            const int64_t C_hash_bits = (C->Y == NULL) ? 0 : (C->Y->vdim - 1) ;
            const int64_t cnvec = C->nvec ;
            int64_t k = GB_hyper_hash_lookup (Ch, cnvec, Cp, C_Yp, C_Yi, C_Yx,
                C_hash_bits, j, &pC_start, &pC_end) ;
            found = (k >= 0) ;
            ASSERT (GB_IMPLIES (found, j == Ch [k])) ;
        }
        else
        { 
            // C is sparse
            pC_start = Cp [j] ;
            pC_end   = Cp [j+1] ;
            found = true ;
        }

        //----------------------------------------------------------------------
        // binary search in kth vector for index i
        //----------------------------------------------------------------------

        if (found)
        { 
            // vector j has been found; now look for index i
            pleft = pC_start ;
            int64_t pright = pC_end - 1 ;

            // Time taken for this step is at most O(log(nnz(C(:,j))).
            const int64_t *restrict Ci = C->i ;
            GB_BINARY_SEARCH_ZOMBIE (i, Ci, pleft, pright, found,
                C->nzombies, is_zombie) ;
        }
    }

    //--------------------------------------------------------------------------
    // set the element
    //--------------------------------------------------------------------------

    if (found)
    {

        //----------------------------------------------------------------------
        // C (i,j) found
        //----------------------------------------------------------------------

        // if not zombie:
        //      no accum:   action: ( =A ): copy A into C
        //      with accum: action: ( C+=A ): accumulate A into C
        // else             action: ( undelete ): bring a zombie back to life

        int8_t cb = (C_is_bitmap) ? C->b [pleft] : 0 ;

        if (!C->iso)
        { 
            void *cx = ((GB_void *) C->x) + (pleft*csize) ;
            if (accum == NULL || is_zombie || (C_is_bitmap && cb == 0))
            { 
                // C(i,j) = (ctype) scalar
                GB_cast_scalar (cx, ccode, scalar, scalar_code, csize) ;
            }
            else
            { 
                // C(i,j) += scalar
                GxB_binary_function faccum = accum->binop_function ;
                ASSERT (faccum != NULL) ;

                GB_cast_function cast_C_to_X, cast_Z_to_Y, cast_Z_to_C ;
                cast_C_to_X = GB_cast_factory (accum->xtype->code, ctype->code);
                cast_Z_to_Y = GB_cast_factory (accum->ytype->code, scalar_code);
                cast_Z_to_C = GB_cast_factory (ctype->code, accum->ztype->code);

                // scalar workspace
                GB_void xaccum [GB_VLA(accum->xtype->size)] ;
                GB_void yaccum [GB_VLA(accum->ytype->size)] ;
                GB_void zaccum [GB_VLA(accum->ztype->size)] ;

                // xaccum = (accum->xtype) cx
                cast_C_to_X (xaccum, cx, ctype->size) ;

                // yaccum = (accum->ytype) scalar
                cast_Z_to_Y (yaccum, scalar, accum->ytype->size) ;

                // zaccum = xaccum "+" yaccum
                faccum (zaccum, xaccum, yaccum) ;

                // cx = (ctype) zaccum
                cast_Z_to_C (cx, zaccum, ctype->size) ;
            }
        }

        if (is_zombie)
        { 
            // bring the zombie back to life
            C->i [pleft] = i ;
            C->nzombies-- ;
        }
        else if (C_is_bitmap)
        { 
            // set the entry in the C bitmap
            C->nvals += (cb == 0) ;
            C->b [pleft] = 1 ;
        }

        return (GrB_SUCCESS) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C (i,j) not found: add a pending tuple
        //----------------------------------------------------------------------

        // action: ( insert )

        // No typecasting can be done.  The new pending tuple must either be
        // the first pending tuple, or its type must match the prior pending
        // tuples.

        //----------------------------------------------------------------------
        // check for wait
        //----------------------------------------------------------------------

        bool wait = false ;

        if (C->Pending == NULL)
        { 
            // the new pending tuple is the first one, so it will define
            // C->type-pending = stype.  No need to wait.
            wait = false ;
        }
        else
        {
            if (stype != C->Pending->type)
            { 
                // the scalar type (stype) must match the type of the
                // prior pending tuples.  If the type is different, prior
                // pending tuples must be assembled first.
                wait = true ;
            }
            else if
            (
                // the types match, now check the pending operator
                ! (
                    // the operators are the same
                    (accum == C->Pending->op)
                    // or both operators are SECOND_Ctype, implicit or explicit
                    || (GB_op_is_second (accum, ctype) &&
                        GB_op_is_second (C->Pending->op, ctype))
                  )
            )
            { 
                wait = true ;
            }
        }

        if (wait)
        { 

            //------------------------------------------------------------------
            // incompatible pending tuples: wait is required
            //------------------------------------------------------------------

            // Pending tuples exist.  Either the pending operator is not
            // SECOND_ctype (implicit or explicit), or the type of prior
            // pending tuples is not the same as the type of the scalar.  This
            // new tuple requires both conditions to hold.  All prior tuples
            // must be assembled before this new one can be added.
            GB_OK (GB_wait (C, "C (setElement:incompatible pending tuples)",
                Werk)) ;

            // repeat the search since the C(i,j) entry may have been in
            // the list of pending tuples.  There are no longer any pending
            // tuples, so this recursion will only happen once.  The
            // pending operator will become the implicit SECOND_ctype, or
            // accum, and the type of the pending tuples will become stype.
            return (GB_setElement (C, accum, scalar, row, col, scalar_code,
                Werk)) ;

        }
        else
        {

            //------------------------------------------------------------------
            // the new tuple is now compatible with prior tuples, if any
            //------------------------------------------------------------------

            ASSERT (GB_PENDING_OK (C)) ;
            ASSERT (GB_ZOMBIES_OK (C)) ;

            // C (i,j) must be added to the list of pending tuples.
            // If this is the first pending tuple, then the type of pending
            // tuples becomes the type of this scalar, and the pending operator
            // becomes NULL, which is the implicit SECOND_ctype operator,
            // or non-NULL if accum is present.
            if (!GB_Pending_add (&(C->Pending), C->iso, (GB_void *) scalar,
                stype, accum, i, j, C->vdim > 1, Werk))
            { 
                // out of memory
                GB_phybix_free (C) ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            ASSERT (GB_PENDING (C)) ;

            // if this was the first tuple, then the pending operator and
            // pending type have been defined
            if (accum == NULL)
            {
                ASSERT (GB_op_is_second (C->Pending->op, ctype)) ;
            }
            else
            {
                ASSERT (C->Pending->op == accum) ;
            }
            ASSERT (C->Pending->type == stype) ;
            ASSERT (C->Pending->size == stype->size) ;

            // one more pending tuple; block if too many of them
            return (GB_block (C, Werk)) ;
        }
    }
}

