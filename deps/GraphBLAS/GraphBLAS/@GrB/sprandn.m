function C = sprandn (arg1, arg2, arg3)
%SPRANDN sparse normally distributed random matrix.
% C = sprandn (A) is a matrix with the same pattern as A,
%   but with normally distributed random entries.
%
% C = sprandn (m,n,d) is a random m-by-n matrix with about m*n*d normally
%   distributed values.  If d == inf, C is a full matrix. To use this
%   function instead of the built-in sprandn, use C = sprandn (m,n,GrB(d)),
%   for example, or C = GrB.random (m,n,d,'normal').
%
% For additional options, see GrB.random.
% The rc parameter for C = sprandn (m,n,d,rc) is not supported.
% C is returned as a double GraphBLAS matrix.
%
% See also GrB/sprandn, GrB/sprandsym, GrB.random.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    % C = sprandn (G)
    G = arg1.opaque ;
    C = GrB (gb_random (G, 'normal')) ;
elseif (nargin == 3)
    % C = sprandn (m, n, d)
    m = gb_get_scalar (arg1) ;
    n = gb_get_scalar (arg2) ;
    d = gb_get_scalar (arg3) ;
    C = GrB (gb_random (m, n, d, 'normal')) ;
else
    % the 'rc' input option is not supported
    error ('GrB:error', 'usage: sprandn(A) or sprandn(m,n,d)') ;
end

