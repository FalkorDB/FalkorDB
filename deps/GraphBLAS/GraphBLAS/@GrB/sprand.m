function C = sprand (arg1, arg2, arg3)
%SPRAND sparse uniformly distributed random matrix.
% C = sprand (A) is a matrix with the same pattern as A, but with
%   uniformly distributed random entries.  This usage is identical to
%   C = GrB.random (A).
%
% C = sprand (m,n,d) is a random m-by-n matrix with about m*n*d uniformly
%   distributed values.  If d == inf, C is a full matrix. To use this
%   function instead of the built-in sprand, use C = sprand (m,n,GrB(d)),
%   for example, or C = GrB.random (m,n,d).
%
% For additional options, see GrB.random.
% The rc parameter for C = sprand (m,n,d,rc) is not supported.
% The entries in C will greater than zero and less than one.
% C is returned as a double GraphBLAS matrix.
%
% See also GrB/sprandn, GrB/sprandsym, GrB.random.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    % C = sprand (G)
    G = arg1.opaque ;
    C = GrB (gb_random (G)) ;
elseif (nargin == 3)
    % C = sprand (m, n, d)
    m = gb_get_scalar (arg1) ;
    n = gb_get_scalar (arg2) ;
    d = gb_get_scalar (arg3) ;
    C = GrB (gb_random (m, n, d)) ;
else
    % the 'rc' input option is not supported
    error ('GrB:error', 'usage: sprand(A) or sprand(m,n,d)') ;
end

