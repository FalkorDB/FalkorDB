// Changes to the API to support 32/64 bit integers inside GrB_Matrix &
// GrB_Vector objects.

// All existing methods will continue to work, for backward compatibility,
// but some GxB methods will be declared "historical" (kept in the library,
// and still working, but no longer documented).

// Last updated: Jan 21, 2025.

//==============================================================================
// get/set
//==============================================================================

// NEW: options for GrB_get/set.

    // GrB_GLOBAL, GrB_Matrix, GrB_Vector, GrB_Scalar: get/set
    GxB_ROWINDEX_INTEGER_HINT           // hint for row indices
    GxB_COLINDEX_INTEGER_HINT           // hint for column indices
    GxB_OFFSET_INTEGER_HINT             // hint for offsets

    // GrB_Matrix, GrB_Vector, GrB_Scalar: get only
    GxB_ROWINDEX_INTEGER_BITS           // # bits for row indices
    GxB_COLINDEX_INTEGER_BITS           // # bits for column indices
    GxB_OFFSET_INTEGER_BITS             // # bits for offsets

// For GrB_GLOBAL, GrB_set can set these values to:

        32  // prefer 32 bit integers; use 64-bit when necessary (default)
        64  // prefer 64 bit integers

// For GrB_Matrix, GrB_Vector, and GrB_Scalar: GrB_set can use:

        0   // use the global default (default)
        32  // prefer 32 bit integers; use 64-bit when necessary
        64  // prefer 64 bit integers

// Examples:

    GrB_set (GrB_GLOBAL, 64, GxB_ROWINDEX_INTEGER_HINT) ;
    GrB_set (A, 64, GxB_ROWINDEX_INTEGER_HINT) ;
    int bits ;  // returned as 32 or 64:
    GrB_get (A, &bits, GxB_ROWINDEX_INTEGER_BITS) ;
    // the following can change the integers held in the matrix A:
    GrB_set (A, 32, GxB_ROWINDEX_INTEGER_HINT) ;

// If a matrix is m-by-n with e entries, with default settings (prefer 32):
//
//      if m > 2^31: use 64-bit integers; else use 32-bits for row indices
//      if n > 2^31: use 64-bit integers; else use 32-bits for col indices
//      if e > 2^32: use 64-bit integers; else use 32-bits for offsets
//
// This gives up to 8 different matrix types when the matrix is hypersparse.
// Sparse matrices use just two of these integer types.  Bitmap and full
// matrices use none.

// All methods that currently take inputs of type (uint64_t *) for list of
// integer row/column indices (GrB_assign, GxB_subassign, GrB_extract,
// GrB_build, GrB_extractTuples) will be augmented to take GrB_Vector inputs
// (I and J, and X for build and extractTuples).  They will be modifiable by
// the descriptor:

// Options for GrB_get/set for GrB_Descriptor only:

    GxB_ROWINDEX_LIST   // defines how the row index list is interpretted
    GxB_COLINDEX_LIST   // defines how the col index list is interpretted
    GxB_VALUE_LIST      // defines how the values list is interpretted
                        // for GrB_build

// Each of settings can take on three different values:

    GxB_USE_VALUES == GrB_DEFAULT     // use the values of the vector (default)
    GxB_USE_INDICES         // use the indices entries in the vector
    GxB_IS_STRIDE           // use the values, of size 3, for lo:inc:hi
                            // I = 0:10:1000

// However, GrB_build will not use GxB_IS_STRIDE, and GrB_extractTuples will
// use none of these descriptor settings.

// The input integer GrB_Vectors I and J can have any built-in data type, but
// they will be typecasted to uint32 (if OK) or uint64 for use inside
// GraphBLAS.  Integer types of uint32, int32, uint64, and int64 will not
// require any typecasting and thus will be fastest, but any other built-in
// type is OK.  The GrB_Vector X for GxB_build is not typecasted.

// For GxB_IS_STRIDE, the GrB_Vector I (or J) must have nvals(I) == 3, and the
// three values are interpretted as (begin,end,stride), for the MATLAB notation
// begin:stride:end.

// The GrB_Vectors I and J for assign, subassign, and extract can have any
// sparsity format, and need not have nvals(I) == length(I).  They can be
// sparse.

//==============================================================================
// GrB_build:
//==============================================================================

// EXISTING methods: use (uint64_t *) arrays for the row/col indices,
// and a C array for X.  The _TYPE suffix is _BOOL, _INT*, _UINT*, _FP* and
// _FC*; these methods will not be extended to use GrB_Vectors for I and J.
// Instead, new methods that use all GrB_Vectors for I,J,X will be added.

// 4 NEW methods: these use GrB_Vectors for all lists: I, J, and X,
// with a suffix "_Vector" added to the name:

GrB_Info GxB_Vector_build_Vector // build a vector from (I,X) tuples
(
    GrB_Vector w,               // vector to build
    const GrB_Vector I_vector,  // row indices
    const GrB_Vector X_vector,  // values
    const GrB_BinaryOp dup,     // binary function to assemble duplicates
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_build_Scalar_Vector // build a vector from (I,s) tuples
(
    GrB_Vector w,               // vector to build
    const GrB_Vector I_vector,  // row indices
    const GrB_Scalar scalar,    // value for all tuples
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_build_Vector // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,               // matrix to build
    const GrB_Vector I_vector,  // row indices
    const GrB_Vector J_vector,  // col indices
    const GrB_Vector X_vector,  // values
    const GrB_BinaryOp dup,     // binary function to assemble duplicates
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_build_Scalar_Vector // build a matrix from (I,J,X) tuples
(
    GrB_Matrix C,               // matrix to build
    const GrB_Vector I_vector,  // row indices
    const GrB_Vector J_vector,  // col indices
    GrB_Scalar scalar,          // value for all tuples
    const GrB_Descriptor desc
) ;

// GrB_Vector_build includes the 2 new GxB_Vector_build*Vector methods:
// GrB_Vector_build_TYPE          (w, I, X, nvals, dup)
// GxB_Vector_build_Scalar        (w, I, s, nvals, dup)
// GxB_Vector_build_Vector        (w, I, X, dup, desc) where I,X are GrB_Vector
// GxB_Vector_build_Scalar_Vector (w, I, s, desc ) where I is GrB_Vector

// GrB_Matrix_build includes the 2 new GxB_Matrix_build*Vector methods:
// GrB_Matrix_build_TYPE          (C, I, J, X, nvals, dup)
// GxB_Matrix_build_Scalar        (C, I, J, s, nvals, dup)
// GxB_Matrix_build_Vector        (C, I, J, X, dup, desc): I,J,X are GrB_Vector
// GxB_Matrix_build_Scalar_Vector (C, I, J, s, desc ): I,J are GrB_Vector

//==============================================================================
// GrB_extractTuples:
//==============================================================================

// 28 EXISTING methods for each of the 14 built-in data types:

GrB_Info GrB_Vector_extractTuples_TYPE      // [I,~,X] = find (v)
GrB_Info GrB_Matrix_extractTuples_TYPE      // [I,J,X] = find (A)

// 2 NEW methods:  all I,J,X are GrB_Vectors.  On output, they are dense
// vectors with nvals(I)=nvals(J)=nvals(X) = length(I)=length(J)=length(X).
// Their GrB_Types are revised to match the internal data types for I, J, and
// X.  This is similar to how dense output vectors are used in the
// GxB_Container by GxB_unload_*_into_Container; see below.

GrB_Info GxB_Vector_extractTuples_Vector    // [I,~,X] = find (v)
(
    GrB_Vector I_vector,    // row indices
    GrB_Vector X_vector,    // values
    const GrB_Vector v,     // vector to extract tuples from
    const GrB_Descriptor desc   // currently unused; for future expansion
) ;

GrB_Info GxB_Matrix_extractTuples_Vector    // [I,J,X] = find (A)
(
    GrB_Vector I_vector,    // row indices
    GrB_Vector J_vector,    // col indices
    GrB_Vector X_vector,    // values
    const GrB_Matrix A,     // matrix to extract tuples from
    const GrB_Descriptor desc   // currently unused; for future expansion
) ;

// The new GxB_*_extractTuples_Vector methods are added to the existing
// polymorphic GrB_Vector_extractTuples and GrB_Matrix_extractTuples:

// GrB_Vector_extractTuple includes:
// GrB_Vector_extractTuples_TYPE   (I, X, nvals, V) where I,X are (uint64_t *)
// GxB_Vector_extractTuples_Vector (I, X, V, d), where I,X are all GrB_Vector

// GrB_Matrix_extractTuple includes:
// GrB_Matrix_extractTuples_TYPE   (I, J, X, nvals, A); I,J,X are (uint64_t *)
// GxB_Matrix_extractTuples_Vector (I, J, X, A, d), where I,J,X are GrB_Vector

//==============================================================================
// GrB_extract:
//==============================================================================

// 3 EXISTING methods, all use (uint64_t *) for I and J:

GrB_Info GrB_Vector_extract         // w<mask> = accum (w, u(I))
GrB_Info GrB_Matrix_extract         // C<M> = accum (C, A(I,J))
GrB_Info GrB_Col_extract            // w<mask> = accum (w, A(I,j))

// 3 NEW methods, using all GrB_Vectors for I and J, with the suffix "_Vector"
// added to the name:

GrB_Info GxB_Vector_extract_Vector  // w<mask> = accum (w, u(I))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector I_vector,      // row indices
    const GrB_Descriptor desc       // descriptor for w and mask
) ;

GrB_Info GxB_Matrix_extract_Vector  // C<M> = accum (C, A(I,J))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Vector I_vector,      // row indices
    const GrB_Vector J_vector,      // column indices
    const GrB_Descriptor desc       // descriptor for C, M, and A
) ;

GrB_Info GxB_Col_extract_Vector     // w<mask> = accum (w, A(I,j))
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Vector I_vector,      // row indices
    GrB_Index j,                    // column index
    const GrB_Descriptor desc       // descriptor for w, mask, and A
) ;

// should I use GrB_Scalar j??  See below:
GrB_Info GxB_Col_extract_Vector     // w<mask> = accum (w, A(I,j))
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Vector I_vector,      // row indices
    GrB_Scalar j,                    // column index
    const GrB_Descriptor desc       // descriptor for w, mask, and A
) ;

// GrB_extract is a polymorphic interface to the following functions, the
// existing ones and the three new ones:
//
// GrB_Vector_extract        (w,m,acc,u,I,ni,d)
// GxB_Vector_extract_Vector (w,m,acc,u,I,d)          where I is a GrB_Vector
// GrB_Col_extract           (w,m,acc,A,I,ni,j,d)
// GxB_Col_extract_Vector    (w,m,acc,A,I,j,d)        where I is a GrB_Vector
// GrB_Matrix_extract        (C,M,acc,A,I,ni,J,nj,d)
// GxB_Matrix_extract_Vector (C,M,acc,A,I,ni,J,nj,d)  where I,J are GrB_Vector

//==============================================================================
// GxB_subassign:
//==============================================================================

// The existing methods will still take only (uint64_t *) arrays; no methods
// that use (uint32_t *) arrays will be added.

// New methods will be added, where all lists I_vector and J_vector are
// GrB_Vectors.  Methods that take 2 lists will require either all (uint64_t *)
// in the existing API, or will require two GrB_Vector inputs.

// No new methods will be added to extend the scalar assignments when the
// scalars are passed as C scalars instead of GrB_Scalars.

// NEW: 6 methods, appending a suffix of _Vector to exising method for each new
// name:

GrB_Info GxB_Vector_subassign_Vector // w(I)<mask> = accum (w(I),u)
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w(I), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w(I),t)
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector I_vector,      // row indices
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_subassign_Vector // C(I,J)<M> = accum (C(I,J),A)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // mask for C(I,J), unused if NULL
    const GrB_BinaryOp accum,       // accum for Z=accum(C(I,J),T)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Vector I_vector,      // row indices
    const GrB_Vector J_vector,      // column indices
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Col_subassign_Vector   // C(I,j)<M> = accum (C(I,j),u)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(I,j), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(I,j),t)
    const GrB_Vector u,             // input vector
    const GrB_Vector I_vector,      // row indices
    GrB_Index j,                    // column index
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Row_subassign_Vector   // C(i,J)<mask'> = accum (C(i,J),u')
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(i,J), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(i,J),t)
    const GrB_Vector u,             // input vector
    GrB_Index i,                    // row index
    const GrB_Vector J_vector,      // column indices
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_subassign_Scalar_Vector   // w(I)<mask> = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    const GrB_Scalar scalar,        // scalar to assign to w(I)
    const GrB_Vector I_vector,      // row indices
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_subassign_Scalar_Vector   // C(I,J)<Mask> = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    const GrB_Scalar scalar,        // scalar to assign to C(I,J)
    const GrB_Vector I_vector,      // row indices
    const GrB_Vector J_vector,      // column indices
    const GrB_Descriptor desc
) ;

// The current GxB_subassign polymorphic function will provide access to all
// non-polymorphic *_subassign* functions.  Below, x is a C scalar and s a
// GrB_Scalar, and TYPE is _BOOL, _INT*, _UINT*, _FP*, _FC* (complex), or _UDT.

// GxB_Vector_subassign_TYPE          (w,m,acc,x,I,ni,d)
// GxB_Vector_subassign_Scalar_Vector (w,m,acc,s,I,d)      I is a GrB_Vector
// GxB_Vector_subassign_Scalar        (w,m,acc,s,I,ni,d)
// GxB_Vector_subassign_Vector        (w,m,acc,u,I,d)      I is a GrB_Vector
// GxB_Vector_subassign               (w,m,acc,u,I,ni,d)

// GxB_Matrix_subassign_TYPE          (C,M,acc,x,I,ni,J,nj,d)
// GxB_Matrix_subassign_Scalar_Vector (C,M,acc,s,I,J,d)    I,J are GrB_Vector
// GxB_Matrix_subassign_Scalar        (C,M,acc,s,I,ni,J,nj,d)
// GxB_Col_subassign                  (C,m,acc,u,I,ni,j,d)
// GxB_Col_subassign_Vector           (C,m,acc,u,I,j,d)    I is a GrB_Vector
// GxB_Row_subassign                  (C,m,acc,u,i,J,nj,d)
// GxB_Row_subassign_Vector           (C,m,acc,u,i,J,d)    J is a GrB_Vector
// GxB_Matrix_subassign_Vector        (C,M,acc,A,I,J,d)    I,J are GrB_Vector
// GxB_Matrix_subassign               (C,M,acc,A,I,ni,J,nj,d)

//==============================================================================
// GrB_assign
//==============================================================================

// Same as GxB_subassign.

// 6 NEW methods:

GrB_Info GxB_Vector_assign_Vector   // w<mask>(I) = accum (w(I),u)
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w(I),t)
    const GrB_Vector u,             // first input:  vector u
    const GrB_Vector I_vector,      // row indices
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_assign_Vector   // C<Mask>(I,J) = accum (C(I,J),A)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),T)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Vector I_vector,      // row indices
    const GrB_Vector J_vector,      // column indices
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Col_assign_Vector      // C<M>(I,j) = accum (C(I,j),u)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(:,j), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(I,j),t)
    const GrB_Vector u,             // input vector
    const GrB_Vector I_vector,      // row indices
    GrB_Index j,                    // column index (should it be GrB_Scalar??)
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Row_assign_Vector      // C<mask'>(i,J) = accum(C(i,j),u')
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // mask for C(i,:), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(i,J),t)
    const GrB_Vector u,             // input vector
    GrB_Index i,                    // row index (should it be GrB_Scalar??)
    const GrB_Vector J_vector,      // column indices
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Vector_assign_Scalar_Vector   // w<mask>(I) = accum (w(I),x)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(w(I),x)
    const GrB_Scalar x,             // scalar to assign to w(I)
    const GrB_Vector I_vector,      // row indices
    const GrB_Descriptor desc
) ;

GrB_Info GxB_Matrix_assign_Scalar_Vector   // C<Mask>(I,J) = accum (C(I,J),x)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C(I,J),x)
    const GrB_Scalar x,             // scalar to assign to C(I,J)
    const GrB_Vector I_vector,      // row indices
    const GrB_Vector J_vector,      // column indices
    const GrB_Descriptor desc
) ;

// The current GrB_assign polymorphic function will provide access to all
// non-polymorphic *_assign* functions.  Below, x is a C scalar and s a
// GrB_Scalar, and TYPE is _BOOL, _INT*, _UINT*, _FP*, _FC* (complex), or _UDT.

// GrB_Vector_assign_TYPE          (w,m,acc,x,I,ni,d)
// GxB_Vector_assign_Scalar_Vector (w,m,acc,s,I,d)      where I is a GrB_Vector
// GrB_Vector_assign_Scalar        (w,m,acc,s,I,ni,d)
// GxB_Vector_assign_Vector        (w,m,acc,u,I,d)      where I is a GrB_Vector
// GrB_Vector_assign               (w,m,acc,u,I,ni,d)

// GrB_Matrix_assign_TYPE          (C,M,acc,x,I,ni,J,nj,d)
// GxB_Matrix_assign_Scalar_Vector (C,M,acc,s,I,J,d)    where I,J are GrB_Vector
// GrB_Matrix_assign_Scalar        (C,M,acc,s,I,ni,J,nj,d)
// GrB_Col_assign                  (C,m,acc,u,I,ni,j,d)
// GxB_Col_assign_Vector           (C,m,acc,u,I,j,d)    where I is a GrB_Vector
// GrB_Row_assign                  (C,m,acc,u,i,J,nj,d)
// GxB_Row_assign_Vector           (C,m,acc,u,i,J,d)    where J is a GrB_Vector
// GxB_Matrix_assign_Vector        (C,M,acc,A,I,J,d)    where I,J are GrB_Vector
// GrB_Matrix_assign               (C,M,acc,A,I,ni,J,nj,d)

//==============================================================================
// GxB_pack/unpack:
//==============================================================================

// No change to the user API, except to enable them to work with internal
// 32-bit integers.  These will still pack/unpack their contents into (uint64_t
// *) user arrays.  If the matrix has 32-bit integers, this will require a
// typecast.  Thus, performance will be degraded for existing user codes that
// expect O(1) time to pack/unpack their matrices/vectors.  The pack/unpack
// methods will still work but will be declared "historical", which means they
// will be kept but will no longer be documented.

// Rather than extend pack/unpack, new methods using the GxB_Container will
// be added (see below), to rapidly move data into/out of a GrB_Matrix or
// GrB_Vector in O(1) time and space.

// Once GraphBLAS v10.0.0 is released, I will revise LAGraph to remove any
// existing uses of pack/unpack (which would still work, just be inefficient if
// the internal data uses 32-bit integers) and replace them with load/unload
// with the new GxB_Container.

//==============================================================================
// GxB_Container
//==============================================================================

// The GxB_Container is a new NON-opaque object that will contain all of the
// data for a GrB_Matrix or GrB_Vector.  It will have some GrB_Vector and
// GrB_Matrix components but they will be opaque since the GrB_Matrix and
// GrB_Vector objects remain opaque.  Since the GxB_Container struct is visible
// to the end user, it will have some extra components for future expansion, in
// case I add new data formats.

// I still need to document each of these components in the Container struct:

struct GxB_Container_struct
{
    // 16 words of uint64_t / int64_t:
    uint64_t nrows ;
    uint64_t ncols ;
    int64_t nrows_nonempty ;
    int64_t ncols_nonempty ;
    uint64_t nvals ;
    uint64_t u64_future [11] ;      // for future expansion

    // 16 words of uint32_t / int32_t:
    int32_t format ;                // GxB_HYPERSPARSE, GxB_SPARSE, GxB_BITMAP,
                                    // or GxB_FULL
    int32_t orientation ;           // GrB_ROWMAJOR or GrB_COLMAJOR
    uint32_t u32_future [14] ;      // for future expansion

    // 16 GrB_Vector objects:
    GrB_Vector p ;
    GrB_Vector h ;
    GrB_Vector b ;
    GrB_Vector i ;
    GrB_Vector x ;
    GrB_Vector vector_future [11] ; // for future expansion

    // 16 GrB_Matrix objects:
    GrB_Matrix Y ;
    GrB_Matrix matrix_future [15] ; // for future expansion

    // 32 words of bool
    bool iso ;
    bool jumbled ;
    bool bool_future [30] ;         // for future expansion

    // 16 (void *) pointers
    void *void_future [16] ;        // for future expansion
} ;

typedef struct GxB_Container_struct *GxB_Container ;

//==============================================================================
// GxB_Container methods
//==============================================================================

// NEW: load/unload into/from a GxB_Container.

// If there is no pending work in the GrB_Matrix A or GrB_Vector V, then
// all unload methods will take O(1) time and NO new space malloc/freed AT ALL
// (not even small constant space).

// All load methods take O(1) time and NO new space malloc'd.  Any prior
// content of the GrB_Matrix V and A is freed, but if they have dimension
// 0-by-0 (for a matrix) or length 0 (for a vector) then no free's will occur.

// When a GrB_Matrix A or GrB_Vector V are "unloaded" into a Container, it
// becomes 0-by-0 with no entries.  Their type is preserved. This is unlike the
// existing pack/unpack, which do not change the dimensions of A and V.

// When a GrB_Matrix A or GrB_Vector V are "loaded" from a Container, their
// dimensions and type are changed to match the data from the Container.

// To support the future SparseBLAS, a GrB_Matrix A or GrB_Vector V can take on
// "readonly" content.  These are pointers user-provided arrays that a moved
// into A or V, but declared "readonly" by GraphBLAS.  Ownership of these
// arrays is kept by the user application.  If GraphBLAS is asked to modify
// a matrix with any readonly content, it will refuse and return an error
// code (info = GxB_OUTPUT_IS_READONLY).

GrB_Info GxB_Container_new (GxB_Container *Container) ;

    // Creates a new container.

GrB_Info GxB_Container_free (GxB_Container *Container) ;

    // Frees a container.

GrB_Info GxB_load_Matrix_from_Container     // GrB_Matrix <- GxB_Container
(
    GrB_Matrix A,               // matrix to load from the Container.  On input,
                                // A is a matrix of any size or type; on output
                                // any prior size, type, or contents is freed
                                // and overwritten with the Container.
    GxB_Container Container,    // Container with contents to load into A
    const GrB_Descriptor desc   // currently unused
) ;

    // Moves data from the Container into the matrix A.

GrB_Info GxB_load_Vector_from_Container     // GrB_Vector <- GxB_Container
(
    GrB_Vector V,               // GrB_Vector to load from the Container
    GxB_Container Container,    // Container with contents to load into A
    const GrB_Descriptor desc   // currently unused
) ;

    // Moves data from the Container into the vector V.

GrB_Info GxB_unload_Matrix_into_Container   // GrB_Matrix -> GxB_Container
(
    GrB_Matrix A,               // matrix to unload into the Container
    GxB_Container Container,    // Container to hold the contents of A
    const GrB_Descriptor desc   // currently unused
) ;

    // Moves data from the matrix A into the Container.

GrB_Info GxB_unload_Vector_into_Container   // GrB_Vector -> GxB_Container
(
    GrB_Vector V,               // vector to unload into the Container
    GxB_Container Container,    // Container to hold the contents of V
    const GrB_Descriptor desc   // currently unused
) ;

    // Moves data from the matrix A into the Container.

// The following two methods move data into/from a single GrB_Vector V.
// For GxB_Vector_unload, the input vector V must have all its entries present.

GrB_Info GxB_Vector_load
(
    // input/output:
    GrB_Vector V,           // vector to load from the C array X
    void **X,               // numerical array to load into V
    // input:
    GrB_Type type,          // type of X
    uint64_t n,             // # of entries in X
    uint64_t X_size,        // size of X in bytes (at least n*(sizeof the type))
    int handling,           // GrB_DEFAULT (0): transfer ownership to GraphBLAS
                            // GxB_IS_READONLY: X treated as readonly;
                            //      ownership kept by the user application
    const GrB_Descriptor desc   // currently unused; for future expansion
) ;

    // GxB_Vector_load moves a C array of length n into a GrB_Vector, which
    // takes on length(V)==n on output, with all entries present (V is dense).
    // Any prior content of V is discarded, except for its user-definable name
    // (via GrB_get/set), and its other get/set options.  The GrB_Vector V will
    // have the type given by the input GrB_Type parameter, which possibily
    // changes the type of V.

    // On output, if handling is GrB_DEFAULT, *X is set to NULL to denote that
    // its content has been moved into V.  If handling is GxB_IS_READONLY, then
    // *X is not changed.

GrB_Info GxB_Vector_unload
(
    // input/output:
    GrB_Vector V,           // vector to unload
    void **X,               // numerical array to unload from V
    // output:
    GrB_Type *type,         // type of X
    uint64_t *n,            // # of entries in X
    uint64_t *X_size,       // size of X in bytes (at least n*(sizeof the type))
    int *handling,          // see GxB_Vector_load
    const GrB_Descriptor desc   // currently unused; for future expansion
) ;

    // GxB_Vector_unload moves the values of V into a C array of length n.
    // On input, the GrB_Vector V must be dense, with nvals(V) == length(V);
    // if nvals(V) < length(V), then this method returns an error (currently
    // GrB_INVALID_OBJECT but perhaps a new error code would be better).
    // On output, the GrB_Vector V has length(V) == 0.  If the vector V is
    // iso-valued on input, it is expanded into a non-iso C array of length n.

//------------------------------------------------------------------------------
// Example
//------------------------------------------------------------------------------

// Usage: with a given GrB_Matrix A to unload/load, of size nrows-by-ncols,
// with nvals entries, of type xtype.  The following will take O(1) time,
// and the only mallocs are in GxB_Container_new (which can be reused for
// an arbitrary number of load/unload cycles), and the only frees are in
// GxB_Container_free.

// Note that getting C arrays from a GrB_Matrix will now be a 2-step process:
// First unload the matrix A into a Container, giving GrB_Vectors Container->p,
// Container->i, Container->x, etc, and then unloading those dense vectors into
// C arrays.  This may seem tedious but it allows everything to be done in
// O(1) time and space (often NO new malloc'd space), and it allows support for
// arbitrary integers for the p, h, and i components of a matrix.

GxB_Container_new (&Container) ;    // requires several O(1)-sized mallocs

// NO malloc/free will occur below, until GxB_Container_free.

for (as many times as you like)
{

    GxB_unload_Matrix_into_Container (A, Container, desc) ;
    // A is now 0-by-0 with nvals(A)=0.  Its type is unchanged.

    // All of the following is optional; if any item in the Container is not
    // needed by the user, it can be left as-is, and then it will be put
    // back into A at the end.  (This is done for the Container->Y).

    // to extract numerical values from the Container:
    void *x = NULL ;
    uint64_t nvals = 0, nheld = 0 ;
    GrB_Type xtype = NULL ;
    int x_handling, p_handling, h_handling, i_handling, b_handling ;
    uint64_t x_size, p_size, h_size, i_size, b_size ;
    GxB_Vector_unload (Container->x, &x, &xtype, &nheld, &x_size, &x_handling,
        desc) ;

    // The C array x now has size nheld and contains the values of the original
    // GrB_Matrix A, with type xtype being the original type of the matrix A.
    // The Container->x GrB_Vector still exists but it now has length 0.
    // If the matrix A was iso-valued, nheld == 1.

    // to extract the sparsity pattern from the Container:
    GrB_Type ptype = NULL, htype = NULL, itype = NULL, btype = NULL ;
    void *p = NULL, *h = NULL, *i = NULL, *b = NULL ;
    uint64_t plen = 0, plen1 = 0, nheld = 0 ;

    switch (Container->format)
    {
        case GxB_HYPERSPARSE :
            // The Container->Y matrix can be unloaded here as well,
            // if desired.  Its use is optional.
            GxB_Vector_unload (Container->h, &h, &htype, &plen, &h_size,
                &h_handling, desc) ;
        case GxB_SPARSE :
            GxB_Vector_unload (Container->p, &p, &ptype, &plen1, &p_size,
                &p_handling, desc) ;
            GxB_Vector_unload (Container->i, &i, &itype, &nvals, &i_size,
                &i_handling, desc) ;
            break ;
        case GxB_BITMAP :
            GxB_Vector_unload (Container->b, &b, &btype, &nheld, &b_size,
                &b_handling, desc) ;
            break ;
    }

    // Now the C arrays (p, h, i, b, and x) are all populated and owned by the
    // user application.  They can be modified here, if desired.  Their C type
    // is (void *), and their actual types correspond to ptype, htype, itype,
    // btype, and xtype).

    // to load them back into A, first load them into the Container->[phbix]
    // vectors:
    switch (Container->format)
    {
        case GxB_HYPERSPARSE :
            // The Container->Y matrix can be loaded here as well,
            // if desired.  Its use is optional.
            GxB_Vector_load (Container->h, &h, htype, plen, h_size,
                h_handling, desc) ;
        case GxB_SPARSE :
            GxB_Vector_load (Container->p, &p, ptype, plen1, p_size,
                p_handling, desc) ;
            GxB_Vector_load (Container->i, &i, itype, nvals, i_size,
                i_handling, desc) ;
            break ;
        case GxB_BITMAP :
            GxB_Vector_load (Container->b, &b, btype, nheld, b_size,
                b_handling, desc) ;
            break ;
    }
    GxB_Vector_load (Container->x, &x, xtype, nheld, x_size,
        x_handling, desc) ;

    // Now the C arrays p, h, i, b, and x are all NULL.  They are in the
    // Container->p,h,b,i,x GrB_Vectors.  Load A from the non-opaque Container:

    GxB_load_Matrix_from_Container (A, Container, desc) ;
    // A is now back to its original state.  The Container and its p,h,b,i,x
    // GrB_Vectors exist but its vectors all have length 0.

}

GxB_Container_free (&Container) ;    // does several O(1)-sized free's

//==============================================================================
// serialize/deserialize:
//==============================================================================

// No change to user-visible API.  The serialized blobs created by GrB v10.0.0
// will be smaller.

// The new serialized blobs will exploit any 32-bit integers in the matrix
// being serialized.  If such a blob is passed to GraphBLAS 9.x or earlier,
// then those older versions of GraphBLAS will safely return an error and
// refuse to deserialize them.

// If GraphBLAS v10 serializes a matrix with all-64-bit integers, then
// GraphBLAS v9.x is able to deserialize it.  The user application can tell
// GrB v10 to use all-64-bits and then serialize it, so it would be able to
// create a backward-compatible serialized blob, if that is desired.

// All serialized blobs created by any prior GraphBLAS version can be
// deserialized by GraphBLAS v10.  Such matrices will necessarily use
// all-64-bit integers, but they will work just fine in GraphBLAS v10.

