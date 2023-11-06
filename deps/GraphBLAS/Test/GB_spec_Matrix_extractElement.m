function [x no_value] = GB_spec_Matrix_extractElement (A, i, j, xclass)
%GB_SPEC_MATRIX_EXTRACTELEMENT a mimic of GrB_Matrix_extractElement

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

A = GB_spec_matrix (A) ;
if (nargin < 4)
    xclass = A.class ;
end

% GraphBLAS indices are zero based, but (i,j) below must be 1-based
i = i+1 ;
j = j+1 ;

no_value = ~(A.pattern (i,j)) ;

if (no_value)
    % The spec says x is not modified, but a built-in function must assign a
    % value to all its outputs.  The mexFunction interface to GraphBLAS
    % also returns zero in this case.
    x = 0 ;
else
    x = A.matrix (i,j) ;
end

x = GB_mex_cast (x, xclass) ;


