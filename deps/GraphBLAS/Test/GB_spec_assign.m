function C = GB_spec_assign (C, Mask, accum, A, I, J, descriptor, scalar)
%GB_SPEC_ASSIGN a mimic of GrB_assign (but not Row or Col variants)
%
% Usage:
% C = GB_spec_assign (C, Mask, accum, A, I, J, descriptor, scalar)
%
% Computes C<Mask>(I,J) = accum(C(I,J),A), in GraphBLAS notation.
%
% This function does the same thing as GrB_Matrix_assign, GrB_Vector_assign,
% GrB_Matrix_assign_TYPE, and GrB_Vector_assign_TYPE.  For these uses, the Mask
% must always be the same size as C.  All of C can be affected (if C_replace is
% true, for example).
%
% This function does not mimic the GrB_Row_assign and GrB_Col_assign functions
% since they behave differently; their Mask is a single row/column, and they do
% not affect any part of C outside that row or column.  Those two functions
% have their own GB_spec_Row_assign.m and GB_spec_Col_assign.m functions.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || nargin ~= 8)
    error ('usage: C = GB_spec_assign (C, Mask, accum, A, I, J, descriptor, scalar)') ;
end

% Convert inputs to dense matrices with explicit patterns and types,
% and with where X(~X.pattern)==identity for all matrices A, B, and C.
C = GB_spec_matrix (C) ;
A = GB_spec_matrix (A) ;
[C_replace Mask_comp Atrans Btrans Mask_struct] = ...
    GB_spec_descriptor (descriptor) ;
Mask = GB_spec_getmask (Mask, Mask_struct) ;

%-------------------------------------------------------------------------------
% do the work via a clean *.m interpretation of the entire GraphBLAS spec
%-------------------------------------------------------------------------------

% apply the descriptor to A
if (Atrans)
    A.matrix = A.matrix.' ;
    A.pattern = A.pattern' ;
end

% expand I and J if empty
if (ischar (I) & isempty (I))
    % I = '' is treated as the empty list
    I = [ ] ;
elseif (isempty (I) || isequal (I, ':'))
    % I = [ ] is treated as ":"
    nrows = size (C.matrix, 1) ;
    I = 1:nrows ;
end
if (ischar (J) & isempty (J))
    % J = '' is treated as the empty list
    J = [ ] ;
elseif (isempty (J) || isequal (J, ':'))
    % J = [ ] is treated as the ":"
    ncols = size (C.matrix, 2) ;
    J = 1:ncols ;
end

if (scalar)
    % scalar expansion: remove duplicates and expand A into a matrix
    I = unique (I) ;
    J = unique (J) ;
    ni = length (I) ;
    nj = length (J) ;
    A.matrix  (1:ni, 1:nj) = A.matrix (1,1) ;
    A.pattern (1:ni, 1:nj) = A.pattern (1,1) ;
end

%-------------------------------------------------------------------------------
% compute the submatrix in Z, using accum
%-------------------------------------------------------------------------------

% initialize Z = C
Z.matrix  = C.matrix ;
Z.pattern = C.pattern ;
Z.class   = C.class ;

% extract the C(I,J) submatrix
S.matrix  = C.matrix  (I,J) ;
S.pattern = C.pattern (I,J) ;
S.class   = C.class ;

% apply the accum operator, ZIJ = accum (S, A)
ZIJ = GB_spec_accum (accum, S, A) ;

% assign the submatrix into Z
Z.matrix  (I,J) = ZIJ.matrix ;
Z.pattern (I,J) = ZIJ.pattern ;

%-------------------------------------------------------------------------------
% apply the Mask and C_replace
%-------------------------------------------------------------------------------

% C<Mask> = Z: apply the Mask and C_replace, and return the result
C = GB_spec_mask (C, Mask, Z, C_replace, Mask_comp, 0) ;

