function C = random (varargin)
%GRB.RANDOM random sparse matrix.
% C = GrB.random (A) has the same pattern as A, but with uniformly
%   distributed random entries.
%
% C = GrB.random (m, n, d) is a random m-by-n GraphBLAS matrix, with
%   about d*m*n uniformly distributed entries.  The entries are
%   constructed by computing d*m*n entries at random positions, and
%   then any duplicates are discarded, so if d is large or m*n is
%   small, then C will have fewer than d*m*n entries.  The value of
%   d may exceed 1 (this differs from the built-in sprand, which limits
%   d to 1).  If d is inf, then C is generated as a full GraphBLAS
%   matrix, with numel (C) = m*n.
%
% Optional parameters may be used, in any order, after the A or m,n,d
% arguments:
%
%   C = GrB.random (..., 'uniform', ...) uses a uniform distribution
%           of values, with entries greater than zero and less than one.
%
%   C = GrB.random (..., 'normal', ...) uses a normal distribution, like
%           the built-in sprandn.
%
%   C = GrB.random (..., 'range', [lo hi], ...) changes the range of
%           the random numbers.  If 'range' is not present, the default
%           is double ([0 1]).  The type of [lo hi] determines the type
%           of the random matrix C.  If [lo hi] is logical, all entries
%           in the pattern are true.  If [lo hi] is 'double' or 'single',
%           then if the random number generator (rand or randn) produces
%           a double random value of x, it is scaled to (hi-lo)*x+lo.  If
%           [lo hi] is integer, then x becomes floor ((hi-lo+1)*x + lo),
%           which is then typecasted to the requested. integer type. This
%           scaling applies to both the 'uniform' and 'normal'
%           distribution.  To construct a random complex matrix, pass in
%           [lo hi] as single complex or double complex.
%
%           With the normal distribution, [lo hi] specifies the mean (lo)
%           and the standard deviation (hi) of the final distribution.
%
%   C = GrB.random (A, 'symmetric', ...) creates a symmetric matrix, like
%           the built-in C = sprandsym (A), except that the default
%           distribution is 'uniform'.  The input matrix A must be
%           square.  Only tril(A) is used to construct C.
%
%   C = GrB.random (n, d, 'symmetric', ...) creates an n-by-n symmetric
%           matrix C, with a uniform distribution of values.  To create a
%           matrix like C = srandsym (n,d) with the built-in sprandym, use
%           C = GrB.random (n, d, 'symmetric', 'normal').  Note that the
%           arguments (m, n, ...) do not appear; just a single
%           dimension (n, ...).
%
%   To construct a Hermitian matrix instead, use 'hermitian' in place of
%   'symmetric'.
%
% The rc option of the built-in sprand and sprandn is not supported.
%
% Example:
%
%   rng ('default')
%   A = sprand (4, 5, 0.5)          % 4-by-5 with at most 10 entries
%   rng ('default')
%   C = GrB.random (4, 5, 0.5)      % same pattern as A
%   isequal (spones (A), spones (C))
%
%   C = GrB.random (C)              % same pattern but newly random values
%
%   C = GrB.random (2, 4, inf)      % 2-by-4 with all 8 entries
%   C = GrB.random (2, 4, 0.5, 'normal')    % like sprandn (2,4,0.5)
%
%   % random 10-by-10 int16 matrix with entries from -3 to 6,
%   % including explicit zeros, with a uniform distribution
%   C = GrB.random (10, 10, 0.5, 'range', int16 ([-3 6]))
%
%   % larger matrix, with normal distribution:
%   C = GrB.random (1000, 1000, 0.5, 'normal', 'range', int16 ([-3 6])) ;
%   [i,j,x] = find (C) ;
%   histogram (x, 'BinMethod', 'integers') ;
%
%   % lots of explicit zeros, since uint8 saturates:
%   C = GrB.random (1000, 1000, 0.5, 'normal', 'range', uint8 ([-3 6])) ;
%   [i,j,x] = find (C) ;
%   histogram (x, 'BinMethod', 'integers') ;
%
%   % uniform distribution across all possible uint8 values
%   C = GrB.random (1000, 1000, 0.5, 'range', uint8 ([0 255])) ;
%   [i,j,x] = find (C) ;
%   histogram (x, 'BinMethod', 'integers') ;
%
%   % large symmetric matrix with normal distribution
%   C = GrB.random (1e6, 1e-5, 'symmetric', 'normal')
%   [i,j,x] = find (C) ;
%   histogram (x)
%
% See also GrB/sprand, GrB/sprandn, GrB/sprandsym.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = GrB (gb_random (varargin {:})) ;

