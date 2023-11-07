function C = GB_spec_Row_assign (C, Mask, accum, A, i, J, descriptor)
%GB_SPEC_ROW_ASSIGN a mimic of GrB_Row_assign
%
% Usage:
% C = GB_spec_Row_assign (C, Mask, accum, A, i, J, descriptor)
%
% Computes C<Mask'>(i,J) = accum(C(i,J),A'), in GraphBLAS notation.
% Both Mask and A must be column vectors of size m, if C is m-by-1
%
% This function does the same thing as GrB_Row_assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || nargin ~= 7)
    error ('usage: C = GB_spec_Row_assign (C, Mask, accum, A, i, J, descriptor)') ;
end

if (length (i) ~= 1)
    error ('i must be a scalar') ;
end

% Convert inputs to dense matrices with explicit patterns and types,
C = GB_spec_matrix (C) ;

% extract the C(i,:) row and transpose it
X.matrix  = C.matrix  (i,:)' ;
X.pattern = C.pattern (i,:)' ;
X.class   = C.class ;

% X<Mask>(J) = accum (X(J),A), on the column vector X
X = GB_spec_assign (X, Mask, accum, A, J, 1, descriptor, 0) ;

% put the C(i,:) row back
C.matrix  (i,:) = X.matrix' ;
C.pattern (i,:) = X.pattern' ;

