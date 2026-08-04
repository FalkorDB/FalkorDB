function C = sprand (varargin)
%SPRAND sparse uniformly distributed random matrix.
% C = sprand (A) is a matrix with the same pattern as A, but with uniformly
%       distributed random entries.  This usage is identical to
%       C = GhB.random (A).
%
% C = sprand (m,n,d) is a random m-by-n matrix with about m*n*d uniformly
%       distributed values.  If d == inf, C is a full matrix. To use this
%       function instead of the built-in sprand, use C = sprand (m,n,GhB(d)),
%       for example, or C = GhB.random (m,n,d).
%
% For additional options, see GhB.random.  The rc parameter for C = sprand
% (m,n,d,rc) is not supported.  The entries in C will greater than zero and
% less than one.  C is returned as a double GraphBLAS GhB matrix.
%
% See also GhB/sprandn, GhB/sprandsym, GhB.random.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_sprand (1, 'uniform', varargin {:}) ;

