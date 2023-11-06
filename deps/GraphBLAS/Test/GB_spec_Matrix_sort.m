function [C,P] = GB_spec_Matrix_sort (op, A, descriptor)
%GB_SPEC_MATRIX_SORT a mimic of GxB_Matrix_sort
%
% Usage:
% [C,P] = GB_spec_Matrix_sort (op, A, descriptor)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 2 || nargin ~= 3)
    error ('usage: [C,P] = GB_spec_Matrix_sort (op, A, descriptor)') ;
end

A = GB_spec_matrix (A) ;
[opname, optype, ztype, xtype, ytype] = GB_spec_operator (op, A.class) ;
[~, ~, Atrans, ~, ~] = GB_spec_descriptor (descriptor) ;

if (isequal (opname, 'lt'))
    direction = 'ascend' ;
elseif (isequal (opname, 'gt'))
    direction = 'descend' ;
else
    error ('unknown order') ;
end

%-------------------------------------------------------------------------------
% do the work via a clean *.m interpretation of the entire GraphBLAS spec
%-------------------------------------------------------------------------------

% create C and P
[m,n] = size (A.matrix) ;
C.matrix = zeros (m, n, A.class) ;
C.pattern = false (m, n) ;
C.class = A.class ;
P.matrix = zeros (m, n, 'int64') ;
P.pattern = false (m, n) ;
P.class = 'int64' ;

if (Atrans)
    % sort each column of A; ignore implicit zeros
    for j = 1:n
        indices = find (A.pattern (:,j)) ;
        values  = A.matrix (indices, j) ;
        T = sortrows ([values indices], { direction, 'ascend'} ) ;
        nvals = length (indices) ;
        C.matrix (1:nvals, j) = T (:,1)     ; C.pattern (1:nvals, j) = true ;
        P.matrix (1:nvals, j) = T (:,2) - 1 ; P.pattern (1:nvals, j) = true ;
    end
else
    % sort each row of A; ignore implicit zeros
    for i = 1:m
        indices = find (A.pattern (i,:))' ;
        values  = A.matrix (i, indices)' ;
        T = sortrows ([values indices], { direction, 'ascend'} ) ;
        nvals = length (indices) ;
        C.matrix (i, 1:nvals) = T (:,1)'     ; C.pattern (i, 1:nvals) = true ;
        P.matrix (i, 1:nvals) = T (:,2)' - 1 ; P.pattern (i, 1:nvals) = true ;
    end
end

