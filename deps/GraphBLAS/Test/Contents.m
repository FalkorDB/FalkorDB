% GraphBLAS Test/ folder: test GraphBLAS
% See the README.txt file for more details.

% Primary functiuns

%   make    - compiles the test interface to GraphBLAS
%   testall - run all GraphBLAS tests

% mimics of GraphBLAS operations:
%
%   GB_spec_Col_assign            - a mimic of GrB_Col_assign
%   GB_spec_Col_extract           - a mimic of GrB_Col_extract
%   GB_spec_Matrix_extract        - a mimic of GrB_Matrix_extract
%   GB_spec_Matrix_extractElement - a mimic of GrB_Matrix_extractElement
%   GB_spec_Row_assign            - a mimic of GrB_Row_assign
%   GB_spec_Vector_extract        - a mimic of GrB_Vector_extract
%   GB_spec_Vector_extractElement - a mimic of GrB_Matrix_extractElement
%   GB_spec_accum                 - mimic of the Z=accum(C,T) operation in GraphBLAS
%   GB_spec_accum_mask            - apply the accumulator and mask
%   GB_spec_apply                 - a mimic of GrB_apply
%   GB_spec_assign                - a mimic of GrB_assign (but not Row or Col variants)
%   GB_spec_build                 - a version of GrB_Matrix_build and GrB_vector_build
%   GB_spec_compare               - compare mimic result with GraphBLAS result
%   GB_spec_descriptor            - return components of a descriptor
%   GB_spec_Matrix_eWiseAdd       - a mimic of GrB_Matrix_eWiseAdd
%   GB_spec_Vector_eWiseAdd       - a mimic of GrB_Vector_eWiseAdd
%   GB_spec_Matrix_eWiseMult      - a mimic of GrB_Matrix_eWiseMult
%   GB_spec_Vector_eWiseMult      - a mimic of GrB_Vector_eWiseMult
%   GB_spec_extractTuples         - a mimic of GrB_*_extractTuples
%   GB_spec_identity              - the additive identity of a monoid
%   GB_spec_kron                  - a mimic of GrB_kronecker
%   GB_spec_mask                  - mimic of GrB_mask
%   GB_spec_matrix                - a mimic that conforms a matrix to the GraphBLAS spec
%   GB_spec_mxm                   - a mimic of GrB_mxm
%   GB_spec_mxv                   - a mimic of GrB_mxv
%   GB_spec_op                    - apply a unary or binary operator
%   GB_spec_operator              - get the contents of an operator
%   GB_spec_opsall                - return a list of all operators, types, and semirings
%   GB_spec_random                - generate random matrix
%   GB_spec_reduce_to_scalar      - a mimic of GrB_reduce (to scalar)
%   GB_spec_reduce_to_vector      - a mimic of GrB_reduce (to vector)
%   GB_spec_resize                - a mimic of GxB_resize
%   GB_spec_select                - a mimic of GxB_select
%   GB_spec_semiring              - create a semiring
%   GB_spec_subassign             - a mimic of GxB_subassign
%   GB_spec_transpose             - a mimic of GrB_transpose
%   GB_spec_vxm                   - a mimic of GrB_vxm
%   GB_spec_Matrix_eWiseUnion     - a mimic of GxB_Matrix_eWiseUnion
%   GB_spec_Matrix_sort           - a mimic of GxB_Matrix_sort
%   GB_spec_Vector_eWiseUnion     - a mimic of GxB_Vector_eWiseUnion
%   GB_spec_Vector_sort           - a mimic of GxB_Vector_sort
%   GB_spec_binop_positional      - compute a binary positional op
%   GB_spec_concat                - a mimic of GxB_Matrix_concat
%   GB_spec_idxunop               - apply an idxunop
%   GB_spec_is_idxunop            - determine if an op is an idxunop
%   GB_spec_is_positional         - determine if an op is positional
%   GB_spec_mdiag                 - a mimic of GxB_Matrix_diag
%   GB_spec_nbits                 - number of bits in an integer type
%   GB_spec_ones                  - all-ones matrix of a given type.
%   GB_spec_select_idxunop        - a mimic of GrB_select
%   GB_spec_split                 - a mimic of GxB_Matrix_split
%   GB_spec_type                  - determine the class of a built-in matrix
%   GB_spec_unop_positional       - compute a unary positional op
%   GB_spec_vdiag                 - a mimic of GxB_Vector_diag
%   GB_spec_zeros                 - all-zero matrix of a given type.
%   GB_spec_getmask               - return the mask, typecasted to logical

%   GB_user_op                    - apply a complex binary and unary operator
%   GB_user_opsall                - return list of complex operators
%   GB_random_mask                - Mask = GB_random_mask (m, n, d, M_is_csc, M_is_hyper)
%   GB_builtin_complex_get        - get the flag that determines the GrB_Type Complex
%   GB_builtin_complex_set        - set a global flag to determine the GrB Complex type 
%   GB_sparsity                   - a string describing the sparsity
%   GB_spok                       - check if a matrix is valid

% Test scripts:

%   test01      - test GraphBLAS error handling
%   test02      - test GrB_*_dup
%   test04      - test and demo for accumulator/mask and transpose
%   test06      - test GrB_mxm on all semirings
%   test09      - test GxB_subassign

%   test10      - test GrB_apply
%   test11      - test GrB_*_extractTuples
%   test14      - test GrB_reduce
%   test17      - test GrB_*_extractElement
%   test18      - test GrB_eWiseAdd, GxB_eWiseUnion, and GrB_eWiseMult
%   test19      - test GxB_subassign and GrB_*_setElement with many pending operations
%   test19b     - test GrB_assign and GrB_*_setElement with many pending operations

%   test21b     - test GrB_assign
%   test23      - test GrB_*_build
%   test29      - GrB_reduce with zombies

%   test53      - test GrB_Matrix_extract
%   test54      - test GB_subref: numeric case with I=lo:hi, J=lo:hi

%   test69      - test GrB_assign with aliased inputs, C<C>(:,:) = accum(C(:,:),C)

%   test74      - test GrB_mxm: all built-in semirings
%   test75b     - GrB_mxm and GrB_vxm on all semirings (shorter test than test75)
%   test76      - test GxB_resize

%   test80      - rerun test06 with different matrices
%   test81      - test GrB_Matrix_extract with index range, stride, & backwards
%   test82      - test GrB_Matrix_extract with index range (hypersparse)
%   test83      - test GrB_assign with J=lo:0:hi, an empty list, and C_replace true
%   test84      - test GrB_assign (row and column with C in CSR/CSC format)

%   test104     - export/import
%   test108     - test boolean monoids
%   test109     - terminal monoid with user-defined type

%   test124     - GrB_extract, trigger case 6
%   test125     - test GrB_mxm: row and column scaling
%   test127     - test GrB_eWiseAdd and GrB_eWiseMult (all types and operators)
%   test128     - test eWiseMult, eWiseAdd, eWiseUnion, special cases

%   test129     - test GxB_select (tril and nonzero, hypersparse)

%   test130     - test GrB_apply (hypersparse cases)
%   test132     - test GrB_*_setElement and GrB_*_*build
%   test133     - test mask operations (GB_masker)
%   test135     - reduce-to-scalar, built-in monoids with terminal values
%   test136     - GxB_subassign, method 08, 09, 11
%   test137     - GrB_eWiseMult with FIRST and SECOND operators
%   test138     - test assign, with coarse-only tasks in IxJ slice
%   test139     - merge sort, special cases

%   test141     - test GrB_eWiseAdd (all types and operators) for dense matrices
%   test142     - test GrB_assign for dense matrices
%   test144     - test GB_cumsum
%   test145     - test dot4
%   test148     - eWiseAdd with aliases

%   test150     - test GrB_mxm with typecasting and zombies (dot3 and saxpy)
%   test151     - test bitwise operators
%   test151b    - test bitshift operators
%   test152     - test C = A+B for dense A, B, and C
%   test154     - test GrB_apply with scalar binding
%   test155     - test GrB_*_setElement and GrB_*_removeElement
%   test156     - test assign C=A with typecasting
%   test157     - test sparsity formats
%   test159     - test dot and saxpy with positional ops

%   testc2      - test complex A*B, A'*B, A*B', A'*B', A+B
%   testc4      - test complex extractElement and setElement
%   testc7      - test complex assign
%   testca      - test complex mxm, mxv, and vxm
%   testcc      - test complex transpose

%   test160     - test GrB_mxm
%   test162     - test C<M>=A*B with very sparse M
%   test165     - test C=A*B' where A is diagonal and B becomes bitmap

%   test172     - eWiseMult with M bitmap/full
%   test173     - test GrB_assign C<A>=A
%   test174     - bitmap assignment, C<!,repl>+=A
%   test176     - test C(I,J)<M,repl> = scalar (method 09, 11), M bitmap
%   test179     - bitmap select

%   test180     - subassign and assign
%   test181     - test transpose with explicit zeros in the Mask
%   test182     - test for internal wait that changes w from sparse/hyper to bitmap/full
%   test183     - test GrB_eWiseMult with a hypersparse mask
%   test184     - test special cases for mxm, transpose, and build
%   test185     - test dot4 for all sparsity formats
%   test186     - test saxpy for all sparsity formats
%   test187     - test dup/assign for all sparsity formats
%   test188     - test concat
%   test189     - test large assignment

%   test191     - test split
%   test192     - test GrB_assign C<C,struct>=scalar
%   test193     - test GxB_Matrix_diag and GrB_Matrix_diag
%   test194     - test GxB_Vector_diag
%   test195     - test all variants of saxpy3
%   test196     - test large hypersparse concat
%   test197     - test large sparse split
%   test199     - test dot2 with hypersparse

%   test200     - test iso full matrix multiply
%   test201     - test iso reduce to vector and reduce to scalar
%   test202     - test iso add and emult
%   test203     - test iso subref
%   test204     - test iso diag
%   test206     - test iso select
%   test207     - test iso subref
%   test208     - test iso apply, bind 1st and 2nd
%   test209     - test iso build

%   test210     - test iso assign25: C<M,struct>=A, C empty, A dense, M structural
%   test211     - test iso assign
%   test212     - test iso mask all zero
%   test213     - test iso assign (method 05d)
%   test214     - test C<M>=A'*B (tricount)
%   test215     - test C<M>=A'*B (dot2, ANY_PAIR semiring)
%   test216     - test C<A>=A, iso case
%   test219     - test reduce to scalar

%   test220     - test mask C<M>=Z, iso case
%   test221     - test C += A where C is bitmap and A is full
%   test222     - test user selectop for iso matrices
%   test223     - test matrix multiply, C<!M>=A*B
%   test224     - unpack/pack
%   test225     - test mask operations (GB_masker)
%   test226     - test kron with iso matrices
%   test227     - test kron
%   test228     - test serialize/deserialize for all sparsity formats
%   test229     - set setElement

%   test230     - test GrB_apply with idxunop
%   test231     - test GrB_select with idxunp
%   test232     - test assign with GrB_Scalar
%   test234     - test GxB_eWiseUnion
%   test235     - test GxB_eWiseUnion and GrB_eWiseAdd
%   test236     - test GxB_Matrix_sort and GxB_Vector_sort
%   test237     - test GrB_mxm (saxpy4)
%   test238     - test GrB_mxm (dot4 and dot2)
%   test239     - test GxB_eWiseUnion

%   test240     - test GrB_mxm: dot4, saxpy4, saxpy5
%   test241     - test GrB_mxm: swap_rule
%   test242     - test GxB_Iterator for matrices
%   test243     - test GxB_Vector_Iterator
%   test244     - test reshape
%   test245     - test colscale (A*D) and rowscale (D*B) with complex types
%   test246     - test GrB_mxm with different kinds of parallelism
%   test247     - test saxpy3 fine-hash method
%   test249     - GxB_Context object tests

%   test250     - basic tests

% Helper functions

%   nthreads_get        - get # of threads and chunk to use in GraphBLAS
%   nthreads_set        - set # of threads and chunk to use in GraphBLAS
%   test10_compare      - check results for test10
%   test_cast           - z = cast (x,type) but handle complex types
%   test_contains       - same as contains (text, pattern)
%   debug_off           - turn off malloc debugging
%   debug_on            - turn on malloc debugging
%   grbinfo             - print info about the GraphBLAS version
%   irand               - construct a random integer matrix 
%   logstat             - run a GraphBLAS test and log the results to log.txt 
%   runtest             - run a single GraphBLAS test
%   stat                - report status of statement coverage and malloc debugging
%   isequal_roundoff    - compare two matrices, allowing for roundoff errors
%   grb_clear_coverage  - clear current statement coverage
%   grb_get_coverage    - return current statement coverage
%   feature_numcores    - determine # of cores the system has
%   jit_reset           - turn off the JIT and then set it back to its original state

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

