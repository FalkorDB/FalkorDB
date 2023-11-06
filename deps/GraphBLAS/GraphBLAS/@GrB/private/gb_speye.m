function C = gb_speye (func, varargin)
%GB_SPEYE Sparse identity matrix, of any type supported by GraphBLAS.
% Implements C = GrB.eye (...) and GrB.speye (...).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% get the size and type
[m, n, type] = gb_parse_args (func, varargin {:}) ;

% construct the m-by-n identity matrix of the given type
m = max (m, 0) ;
n = max (n, 0) ;
mn = min (m, n) ;
I = int64 (0) : int64 (mn-1) ;
desc.base = 'zero-based' ;

if (isequal (type, 'single complex'))
    X = complex (ones (mn, 1, 'single')) ;
elseif (gb_contains (type, 'complex'))
    X = complex (ones (mn, 1, 'double')) ;
else
    X = ones (mn, 1, type) ;
end

C = gbbuild (I, I, X, m, n, '1st', type, desc) ;

