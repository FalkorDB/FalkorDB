function s = tricount (varargin)
%GRB.TRICOUNT count triangles in a matrix.
% s = GrB.tricount (A) is the number of triangles in the matrix A.  spones (A)
% must be symmetric; results are undefined if spones (A) is unsymmetric.
% Diagonal entries are ignored.
%
% To check the input matrix A, use GrB.tricount (A, 'check').  This check takes
% additional time so by default the input is not checked.
%
% If d is a vector of length n with d(i) equal to the degree of node i, then s
% = tricount (A, d) can be used.  Otherwise, tricount must compute the degrees
% first.
%
% See also GrB.ktruss, GrB.entries.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = GhB.tricount (varargin {:}) ;

