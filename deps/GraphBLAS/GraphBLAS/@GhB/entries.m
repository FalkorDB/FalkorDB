function result = entries (A, varargin)
%GHB.ENTRIES count or query the entries of a matrix.
% An entry A(i,j) in a GraphBLAS matrix is one that is present in the data
% structure.  Unlike a built-in sparse matrix, a GraphBLAS matrix can contain
% explicit zero entries.  All entries in a built-in sparse matrix are nonzero.
% A built-in full matrix has all of its entries present, regardless of their
% value.  The GhB.entries function looks only at the pattern of A, not its
% values.  To exclude explicit entries with a value of zero (or any specified
% additive identity value) use GhB.nonz instead.
%
% e = GhB.entries (A)         number of entries
% e = GhB.entries (A, 'all')  number of entries
% e = GhB.entries (A, 'row')  number of rows with at least one entry
% e = GhB.entries (A, 'col')  number of columns with at least one entry
%
% X = GhB.entries (A, 'list')         list of values of unique entries
% X = GhB.entries (A, 'all', 'list')  list of values of unique entries
% I = GhB.entries (A, 'row', 'list')  list of rows with at least one entry
% J = GhB.entries (A, 'col', 'list')  list of cols with at least one entry
%
% d = GhB.entries (A, 'row', 'degree')
%   If A is m-by-n, then d is a sparse column vector of size m, with d(i) equal
%   to the number of entries in A(i,:).  If A(i,:) has no entries, then d(i) is
%   an implicit zero, not present in the pattern of d, so that I = find (d) is
%   the same I = GhB.entries (A, 'row', 'list').
%
% d = GhB.entries (A, 'col', 'degree')
%   If A is m-by-n, d is a sparse column vector of size n, with d(j) equal to
%   the number of entries in A(:,j).  If A(:,j) has no entries, then d(j) is an
%   implicit zero, not present in the pattern of d, so that I = find (d) is the
%   same I = GhB.entries (A, 'col', 'list').
%
% The result is a built-in scalar or vector, except for the 'degree' usage, in
% which case the result is a GhB vector d.
%
% Example:
%
%   A = magic (5) ;
%   A (A < 10) = 0             % built-in full matrix with some explicit zeros
%   GhB.entries (A)            % all entries present in a built-in full matrix
%   G = GhB (A)                % contains explicit zeros
%   GhB.entries (G)
%   G (A > 18) = sparse (0)    % entries A>18 deleted, has explicit zeros
%   GhB.entries (G)
%   GhB.entries (G, 'list')
%   S = double (G)             % built-in sparse matrix; no explicit zeros
%   GhB.entries (S)
%   GhB.entries (S, 'list')
%
% See also GhB.nonz, nnz, GhB/nnz, nonzeros, GhB/nonzeros.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

result = gb_entries (1, A, varargin {:}) ;

