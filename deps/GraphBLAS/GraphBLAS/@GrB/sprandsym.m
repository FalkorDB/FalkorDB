function C = sprandsym (arg1, arg2)
%SPRANDSYM random symmetric matrix.
% C = sprandsym (A) is a symmetric random matrix.  Its lower triangle and
%   diagonal have the same pattern as tril (A).  The values of C have a
%   normal distribution.  A must be square.  This usage is the same as
%   C = GrB.random (A, 'symmetric', 'normal').
%
% C = sprandsym (n,d) is an n-by-n symmetric random matrix with about n*n*d
%   entries, with a normal distribution.  If d == inf, C is full.  To use
%   this function instead of the built-in sprandsym, use
%   C = sprandsym (n,GrB(d)), or C = GrB.random (n,d,'symmetric','normal').
%
% For additional options, see GrB.random.
% The C = sprandsym (n,d,rc) syntax is not supported.
% C is returned as a double GraphBLAS matrix.
%
% Example:
%
%   A = sprand (1000, 1000, 0.5) ;
%   G = GrB (A) ;
%   C0 = sprandsym (A) ;                % the built-in sprandsym
%   C1 = sprandsym (G) ;                % GrB/sprandsym
%
% See also GrB/sprand, GrB/sprandn, GrB.random.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    % C = sprandsym (G)
    G = arg1.opaque ;
    C = GrB (gb_random (G, 'symmetric', 'normal')) ;
else
    % C = sprandsym (n, d)
    n = gb_get_scalar (arg1) ;
    d = gb_get_scalar (arg2) ;
    C = GrB (gb_random (n, d, 'symmetric', 'normal')) ;
end

