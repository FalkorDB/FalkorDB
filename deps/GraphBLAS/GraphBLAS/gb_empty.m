function C = gb_empty (ghb, varargin)
%GB_EMPTY implements GrB.empty and GhB.empty.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
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

C = gzb (ghb, m, n) ;

