function [C,P] = GB_spec_Vector_sort (op, A, descriptor)
%GB_SPEC_VECTOR_SORT a mimic of GxB_Vector_sort
%
% Usage:
% [C,P] = GB_spec_Vector_sort (op, A, descriptor)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 2 || nargin ~= 3)
    error ('usage: [C,P] = GB_spec_Vector_sort (op, A, descriptor)') ;
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
if (n ~= 1)
    error ('A must be a vector') ;
end
C.matrix = zeros (m, n, A.class) ;
C.pattern = false (m, n) ;
C.class = A.class ;
P.matrix = zeros (m, n, 'int64') ;
P.pattern = false (m, n) ;
P.class = 'int64' ;

    % sort the vector A; ignore implicit zeros
    for j = 1:n
        indices = find (A.pattern (:,j)) ;
        values  = A.matrix (indices, j) ;
        T = sortrows ([values indices], { direction, 'ascend'} ) ;
        nvals = length (indices) ;
        C.matrix (1:nvals, j) = T (:,1)     ; C.pattern (1:nvals, j) = true ;
        P.matrix (1:nvals, j) = T (:,2) - 1 ; P.pattern (1:nvals, j) = true ;
    end


