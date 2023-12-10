function C = empty (varargin)
%GRB.EMPTY construct an empty GraphBLAS sparse matrix.
% C = GrB.empty is a 0-by-0 empty matrix.
% C = GrB.empty (m) is an m-by-0 empty matrix.
% C = GrB.empty ([m n]) or GrB.empty (m,n) is an m-by-n empty matrix,
% where one of m or n must be zero.
%
% All matrices are constructed with the 'double' type.  Use GrB (m,n,type)
% to construct empty matrices of with different types.
%
% See also GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    m = 0 ;
    n = 0 ;
else
    [m, n] = gb_parse_dimensions (varargin {:}) ;
    m = max (m, 0) ;
    n = max (n, 0) ;
    if (~ ((m == 0) || (n == 0)))
        error ('GrB:error', 'at least one dimension must be zero') ;
    end
end

C = GrB (gbnew (m, n)) ;

