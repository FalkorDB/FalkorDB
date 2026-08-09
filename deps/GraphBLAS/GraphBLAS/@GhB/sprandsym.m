function C = sprandsym (varargin)
%SPRANDSYM random symmetric matrix.
% C = sprandsym (A) is a symmetric random matrix.  Its lower triangle and
%       diagonal have the same pattern as tril (A).  The values of C have a
%       normal distribution.  A must be square.  This usage is the same as
%       C = GhB.random (A, 'symmetric', 'normal').
%
% C = sprandsym (n,d) is an n-by-n symmetric random matrix with about n*n*d
%       entries, with a normal distribution.  If d == inf, C is full.  To use
%       this function instead of the built-in sprandsym, use
%       C = sprandsym (n,GhB(d)), or C = GhB.random (n,d,'symmetric','normal').
%
% For additional options, see GhB.random.  The C = sprandsym (n,d,rc) syntax is
% not supported.  C is returned as a double GraphBLAS GhB matrix.
%
% Example:
%
%   A = sprand (1000, 1000, 0.5) ;
%   G = GhB (A) ;
%   C0 = sprandsym (A) ;                % the built-in sprandsym
%   C1 = sprandsym (G) ;                % GhB/sprandsym
%
% See also GhB/sprand, GhB/sprandn, GhB.random.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_sprandsym (1, varargin {:}) ;

