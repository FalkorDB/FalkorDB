function C = GB_spec_Col_assign (C, Mask, accum, A, I, j, descriptor)
%GB_SPEC_COL_ASSIGN a mimic of GrB_Col_assign
%
% Usage:
% C = GB_spec_Col_assign (C, Mask, accum, A, I, j, descriptor)
%
% Computes C<Mask>(I,j) = accum(C(I,j),A), in GraphBLAS notation.
%
% This function does the same thing as GrB_Col_assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || nargin ~= 7)
    error ('usage: C = GB_spec_Col_assign (C, Mask, accum, A, I, j, descriptor)') ;
end

if (length (j) ~= 1)
    error ('j must be a scalar') ;
end

% Convert inputs to dense matrices with explicit patterns and types,
C = GB_spec_matrix (C) ;

% extract the C(:,j) column
X.matrix  = C.matrix  (:,j) ;
X.pattern = C.pattern (:,j) ;
X.class   = C.class ;

% X<Mask>(I) = accum (X(I),A)
X = GB_spec_assign (X, Mask, accum, A, I, 1, descriptor, 0) ;

% put the C(:,j) colum back
C.matrix  (:,j) = X.matrix ;
C.pattern (:,j) = X.pattern ;

