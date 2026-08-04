function result = nonz (A, varargin)
%GHB.NONZ count or query the nonzeros of a matrix.
% A GraphBLAS matrix can include explicit entries that have the value zero.
% These entries never appear in a built-in sparse matrix.  This function counts
% or queries the nonzeros of matrix, checking their value and treating explicit
% zeros the same as entries that do not appear in the pattern of A.
%
% e = GhB.nonz (A)         number of nonzeros
% e = GhB.nonz (A, 'all')  number of nonzeros
% e = GhB.nonz (A, 'row')  number of rows with at least one nonzero
% e = GhB.nonz (A, 'col')  number of columns with at least one nonzero
%
% X = GhB.nonz (A, 'list')         list of values of unique nonzeros
% X = GhB.nonz (A, 'all', 'list')  list of values of unique nonzeros
% I = GhB.nonz (A, 'row', 'list')  list of rows with at least one nonzero
% J = GhB.nonz (A, 'col', 'list')  list of cols with at least one nonzero
%
% d = GhB.nonz (A, 'row', 'degree')
%   If A is m-by-n, then d is a sparse column vector of size m, with d(i) equal
%   to the number of nonzeros in A(i,:).  If A(i,:) has no nonzeros, then d(i)
%   is an implicit zero, not present in the pattern of d, so I = find (d) is
%   the same I = GhB.nonz (A, 'row', 'list').
%
% d = GhB.nonz (A, 'col', 'degree')
%   If A is m-by-n, d is a sparse column vector of size n, with d(j) equal to
%   the number of nonzeros in A(:,j).  If A(:,j) has no nonzeros, then d(j) is
%   an implicit zero, not present in the pattern of d, so I = find (d) is the
%   same I = GhB.nonz (A, 'col', 'list').
%
% With an optional scalar argument as the last argument, the value of the
% 'zero' can be specified; d = GhB.nonz (A, ..., id).  For example, to count
% all entries in A not equal to one, use GhB.nonz (A, 1).
%
% The result is a built-in scalar or vector, except for the 'degree' usage, in
% which case the result is a GhB vector d.
%
% Example:
%
%   A = magic (5) ;
%   A (A < 10) = 0              % built-in full matrix with explicit zeros
%   nnz (A)
%   GhB.nonz (A)                % same as nnz (A)
%   G = GhB (A)                 % contains explicit zeros
%   GhB.nonz (G)                % same as nnz (A)
%   G (A > 18) = sparse (0)     % entries A>18 deleted, explicit zeros
%   GhB.nonz (G)
%   GhB.nonz (G, 'list')
%   S = double (G)              % built-in sparse matrix; no explicit zeros
%   GhB.nonz (S)
%   GhB.nonz (S, 'list')
%
% See also GhB.entries, GhB/nnz, GhB/nonzeros, GhB.prune.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

result = gb_nonz (1, A, varargin {:}) ;

