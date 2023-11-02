function s = isfull (A)
%GRB.ISFULL determine if all entries are present.
% For either a GraphBLAS or built-in matrix, GrB.isfull (A) is true if
% numel(A) == nnz(A).  GrB.isfull (A) is always true if A is a GraphBLAS
% or built-in full matrix.
%
% See also GrB/issparse, GrB/full.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    % GraphBLAS matrix
    A = A.opaque ;
    s = gb_isfull (A) ;
elseif (issparse (A))
    % built-in sparse matrix
    s = (numel (A) == nnz (A)) ;
else
    % built-in full matrix, string, struct, etc
    s = true ;
end

