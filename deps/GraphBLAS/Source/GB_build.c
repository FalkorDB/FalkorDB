//------------------------------------------------------------------------------
// GB_build: build a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CALLED BY: GrB_Matrix_build_*, GrB_Vector_build_*,
//            GxB_Matrix_build_Scalar, GxB_Vector_build_Scalar
// CALLS:     GB_builder

// GrB_Matrix_build_* and GrB_Vector_build_* always build a non-iso matrix.
// X is non-iso, and T is non-iso on output.  dup can be a valid binary
// operator, NULL, or the special binary operator GxB_IGNORE_DUP.
// For a non-iso build, dup may be NULL, which indicates that duplicates are
// not expected.  If any duplicates do appear, an error is reported.
// If dup is GxB_IGNORE_DUP, then duplicates are discarded.  If (i,j,x1)
// appears as the (k1)th tuple, and (i,j,x2) as the (k2)th tuple, and
// k1 < k2, then the first tuple is ignored and C(i,j) is equal to x2.

// GxB_Matrix_build_Scalar and GxB_Vector_build_Scalar always build an iso
// matrix:  X is iso, only the first entry X [0] is used, and dup does not
// appear (it is passed here as GxB_IGNORE_DUP).  The matrix T is iso on
// output.  For an iso build, duplicates are discarded.

// For all of these cases, the tuples must be checked for duplicates.  They
// might be sorted on input, so this condition is checked and exploited if that
// condition is found.  All of these conditions are checked in GB_builder.

// GB_build constructs a matrix C from a list of indices and values.  Any
// duplicate entries with identical indices are assembled using the binary dup
// operator provided on input, or discarded if dup is NULL or GxB_IGNORE_DUP.
// All three types (x,y,z for z=dup(x,y)) must be identical.  The types of dup,
// X, and C must all be compatible.

// Duplicates are assembled using T(i,j) = dup (T (i,j), X (k)) into a
// temporary matrix T that has the same type as the dup operator.  The
// GraphBLAS spec requires dup to be associative so that entries can be
// assembled in any order.  There is no way to check this condition if dup is a
// user-defined operator.  It could be checked for built-in operators, but the
// GraphBLAS spec does not require this condition to cause an error so that is
// not done here.  If dup is not associative, the GraphBLAS spec states that
// the results are not defined.

// SuiteSparse:GraphBLAS provides a well-defined order of assembly, however.
// For a CSC format, entries in [I,J,X] are first sorted in increasing order of
// row and column index via a stable sort, with ties broken by the position of
// the tuple in the [I,J,X] list.  If duplicates appear, they are assembled in
// the order they appear in the [I,J,X] input.  That is, if the same indices i
// and j appear in positions k1, k2, k3, and k4 in [I,J,X], where k1 < k2 < k3
// < k4, then the following operations will occur in order:

//      T (i,j) = X (k1) ;
//      T (i,j) = dup (T (i,j), X (k2)) ;
//      T (i,j) = dup (T (i,j), X (k3)) ;
//      T (i,j) = dup (T (i,j), X (k4)) ;

// This is a well-defined order but the user should not depend upon it since
// the GraphBLAS spec does not require this ordering.  Results may differ in
// different implementations of GraphBLAS.

// However, with this well-defined order, the SECOND operator will result in
// the last tuple overwriting the earlier ones.  This is relied upon internally
// by GB_wait.

// After the matrix T is assembled, it is typecasted into the type of C, the
// final output matrix.  No typecasting is done during assembly of duplicates,
// since mixing the two can break associativity and lead to unpredictable
// results.  Note that this is not the case for GB_wait, which must typecast
// each tuple into its output matrix in the same order they are seen in the
// [I,J,X] pending tuples.

// On input, C must not be NULL.  C->type, C->vlen, C->vdim and C->is_csc must
// be valid on input and are unchanged on output.  C must not have any existing
// entries on input (GrB_*_nvals (C) must return zero, per the specification).
// However, all existing content in C is freed.

// The list of numerical values is given by the void * X array and its type,
// xtype.  The latter is defined by the actual C type of the X parameter in
// the user-callable functions.  However, for user-defined types, there is no
// way of knowing that the X array has the same type as dup or C, since in that
// case X is just a void * pointer.  Behavior is undefined if the user breaks
// this condition.

// C is returned as hypersparse or non-hypersparse, depending on the number of
// non-empty vectors of C.  If C has very few non-empty vectors, then it is
// returned as hypersparse.  Only if the number of non-empty vectors is
// Omega(nh) is C returned as non-hypersparse, which implies nvals is Omega(n),
// where n = # of columns of C if CSC, or # of rows if CSR.  As a result, the
// time taken by this function is just O(nvals*log(nvals)), regardless of what
// format C is returned in.

// The input arrays I, J, and X are not modified.

#define GB_FREE_ALL GrB_Matrix_free (&T) ;
#include "GB_build.h"

GrB_Info GB_build               // build matrix
(
    GrB_Matrix C,               // matrix to build
    const GrB_Index *I,         // row indices of tuples
    const GrB_Index *J,         // col indices of tuples (NULL for vector)
    const void *X,              // values, size 1 if iso
    const GrB_Index nvals,      // number of tuples
    const GrB_BinaryOp dup,     // binary op to assemble duplicates (or NULL)
    const GrB_Type xtype,       // type of X array
    const bool is_matrix,       // true if C is a matrix, false if GrB_Vector
    const bool X_iso,           // if true the C is iso and X has size 1 entry
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // check C
    GrB_Info info ;
    ASSERT (C != NULL) ;
    if (GB_nnz (C) > 0 || GB_PENDING (C))
    { 
        // The matrix has existing entries.  This is required by the GraphBLAS
        // API specification to generate an error, so the test is made here.
        // However, any existing content is safely freed immediately below, so
        // this test is not required, except to conform to the spec.  Zombies
        // are excluded from this test.
        GB_ERROR (GrB_OUTPUT_NOT_EMPTY,
            "Output already has %s", "existing entries") ;
    }

    // check I
    GB_RETURN_IF_NULL (I) ;
    if (I == GrB_ALL)
    { 
        GB_ERROR (GrB_INVALID_VALUE, "Row indices cannot be %s", "GrB_ALL") ;
    }

    // check J
    if (is_matrix)
    {
        GB_RETURN_IF_NULL (J) ;
        if (J == GrB_ALL)
        { 
            GB_ERROR (GrB_INVALID_VALUE, "Column indices cannot be %s",
                "GrB_ALL") ;
        }
    }
    else
    { 
        // only G*B_Vector_build_* calls this function with J == NULL
        ASSERT (J == NULL) ;
    }

    // check X
    GB_RETURN_IF_NULL (X) ;

    if (nvals == GxB_RANGE || nvals == GxB_STRIDE || nvals == GxB_BACKWARDS)
    { 
        GB_ERROR (GrB_INVALID_VALUE, "nvals cannot be %s",
            "GxB_RANGE, GxB_STRIDE, or GxB_BACKWARDS") ;
    }

    if (nvals > GB_NMAX)
    { 
        // problem too large
        GB_ERROR (GrB_INVALID_VALUE, "Problem too large: nvals " GBu
            " exceeds " GBu, nvals, GB_NMAX) ;
    }

    //--------------------------------------------------------------------------
    // check the types
    //--------------------------------------------------------------------------

    // C and X must be compatible
    if (!GB_Type_compatible (xtype, C->type))
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Value(s) of type [%s] cannot be typecast to matrix of type"
            " [%s]\n", xtype->name, C->type->name) ;
    }

    //--------------------------------------------------------------------------
    // check the dup operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp dup2 ;
    bool discard_duplicates = (dup == NULL || dup == GxB_IGNORE_DUP) ;

    if (discard_duplicates)
    { 

        //----------------------------------------------------------------------
        // discard all duplicates
        //----------------------------------------------------------------------

        dup2 = NULL ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // "sum" up duplicates using the binary dup operator
        //----------------------------------------------------------------------

        dup2 = dup ;
        GB_RETURN_IF_FAULTY (dup) ;

        if (GB_OP_IS_POSITIONAL (dup))
        { 
            // dup operator cannot be a positional op
            GB_ERROR (GrB_DOMAIN_MISMATCH, "Positional op z=%s(x,y) "
                "not supported as dup op\n", dup->name) ;
        }

        ASSERT_BINARYOP_OK (dup, "dup for assembling duplicates", GB0) ;

        // check types of dup
        if (dup->xtype != dup->ztype || dup->ytype != dup->ztype)
        { 
            // all 3 types of z = dup (x,y) must be the same.  dup must also be
            // associative but there is no way to check this in general.
            GB_ERROR (GrB_DOMAIN_MISMATCH, "All domains of dup "
                "operator for assembling duplicates must be identical.\n"
                "operator is: [%s] = %s ([%s],[%s])", dup->ztype->name,
                dup->name, dup->xtype->name, dup->ytype->name) ;
        }

        // the type of C and dup must be compatible
        if (!GB_Type_compatible (C->type, dup->ztype))
        { 
            GB_ERROR (GrB_DOMAIN_MISMATCH,
                "Operator [%s] for assembling duplicates has type [%s],\n"
                "cannot be typecast to entries in output of type [%s]",
                dup->name, dup->ztype->name, C->type->name) ;
        }
    }

    //--------------------------------------------------------------------------
    // free all content of C
    //--------------------------------------------------------------------------

    // the type, dimensions, hyper_switch, bitmap_switch and sparsity control
    // are still preserved in C.
    GB_phybix_free (C) ;

    //--------------------------------------------------------------------------
    // build the matrix T
    //--------------------------------------------------------------------------

    // T is always built as hypersparse.  Its type is the same as the z output
    // of the z=dup(x,y) operator if dup is present, or xtype if dup is NULL.
    // If C->type differs from T->type, it is typecasted by
    // GB_transplant_conform.

    // X must be treated as read-only, so GB_builder is not allowed to
    // transplant it into T->x.

    int64_t *no_I_work = NULL ; size_t I_work_size = 0 ;
    int64_t *no_J_work = NULL ; size_t J_work_size = 0 ;
    GB_void *no_X_work = NULL ; size_t X_work_size = 0 ;
    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;
    GB_CLEAR_STATIC_HEADER (T, &T_header) ;
    GrB_Type ttype = (discard_duplicates) ? xtype : dup->ztype ;

    GB_OK (GB_builder (
        T,              // create T using a static header
        ttype,          // the type of T
        C->vlen,        // T->vlen = C->vlen
        C->vdim,        // T->vdim = C->vdim
        C->is_csc,      // T has the same CSR/CSC format as C
        &no_I_work,     // I_work_handle, not used here
        &I_work_size,
        &no_J_work,     // J_work_handle, not used here
        &J_work_size,
        &no_X_work,     // X_work_handle, not used here
        &X_work_size,
        false,          // known_sorted: not yet known
        false,          // known_no_duplicates: not yet known
        0,              // I_work, J_work, and X_work not used here
        is_matrix,      // true if T is a GrB_Matrix
        (int64_t *) ((C->is_csc) ? I : J),  // size nvals
        (int64_t *) ((C->is_csc) ? J : I),  // size nvals, or NULL for vector
        (const GB_void *) X,                // values, size nvals or 1 if iso
        X_iso,          // true if X is iso
        nvals,          // number of tuples
        dup2,           // operator to assemble duplicates (may be NULL)
        xtype,          // type of the X array
        true,           // burble is OK
        Werk
    )) ;

    //--------------------------------------------------------------------------
    // return an error if any duplicates found when they were not expected
    //--------------------------------------------------------------------------

    int64_t tnvals = GB_nnz (T) ;
    if (dup == NULL && nvals != tnvals)
    { 
        // T has been successfully built by ignoring the duplicate values, via
        // the implicit SECOND dup operator.  If the # of entries in T does not
        // match nvals, then duplicates have been detected.  In the v2.0 C API,
        // this is an error condition.  If the user application wants the C
        // matrix returned with duplicates discarded, use dup = GxB_IGNORE_DUP
        // instead. 
        GB_FREE_ALL ;
        GB_ERROR (GrB_INVALID_VALUE, "Duplicates appear (" GBd ") but dup "
            "is NULL", ((int64_t) nvals) - tnvals) ;
    }

    //--------------------------------------------------------------------------
    // determine if T is iso, for non-iso build
    //--------------------------------------------------------------------------

    // GxB_Matrix_build_Scalar and GxB_Vector_build_Scalar always build an iso
    // matrix T, so this test is skipped (X_iso is true in that case).
    // GrB_Matrix_build_[TYPE] and GrB_Vector_build_[TYPE] may have just
    // created an iso-valued matrix T, but this is not yet known.  X_iso is
    // false for these methods.  Since it has not yet been conformed to its
    // final sparsity structure, the matrix T is hypersparse, not bitmap.  It
    // has no zombies or pending tuples, so GB_check_if_iso does need to handle
    // those cases.  T->x [0] is the new iso value of T.

    if (!X_iso && GB_check_if_iso (T))
    { 
        // All entries in T are the same; convert T to iso
        GBURBLE ("(post iso) ") ;
        T->iso = true ;
        GB_OK (GB_convert_any_to_iso (T, NULL)) ;
    }

    //--------------------------------------------------------------------------
    // transplant and typecast T into C, conform C, and free T
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_HYPERSPARSE (T)) ;
    ASSERT (!GB_ZOMBIES (T)) ;
    ASSERT (!GB_JUMBLED (T)) ;
    ASSERT (!GB_PENDING (T)) ;
    return (GB_transplant_conform (C, C->type, &T, Werk)) ;
}

