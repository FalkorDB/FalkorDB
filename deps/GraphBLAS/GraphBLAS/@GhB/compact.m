function [C, I, J] = compact (A, id, symmetric)
%GHB.COMPACT remove empty rows and columns from a matrix.
% C = GhB.compact (A) returns rows and columns from A that have no entries.  It
% has no effect on a full matrix, except to convert it to a GraphBLAS matrix,
% since all entries are present in a full matrix.
%
% Explicit zeros in A are treated as entries, and are not removed by default.
% To remove rows and columns with no entries or only explicit zero entries, use
% C = GhB.compact (A,0).  For a sparse matrix, GhB.compact (A,0) and
% GhB.compact (A) are identical.
%
% To remove rows and colums that either have no entries, or that only have
% entries equal to a particular scalar value, use C = GhB.compact (A, id),
% where id is the scalar value.  To skip this option, use id = [ ].
%
% With two additional output arguments, [C,I,J] = GhB.compact (A, ...), the
% indices of non-empty rows and columns of A are returned, so that C = A (I,J).
% The lists I and J are returned in sorted order.
%
% A third parameter specifies if C should be constructed symmetrically.  C =
% GhB.compact (A, [ ], 'symmetric') constructs a list I = union (I,J), and then
% extracts C = A (I,I).  To use this option, A must be square, but it need not
% be symmetric.  This option is useful for pruning nodes of a directed graph A
% that have no incoming or outgoing edges.  The graph of C will include any
% node with either incoming or outgoing (or both) edges.
%
% Example:
%
%   n = 2^40 ;
%   H = GhB (n,n) ;                 % create a huge hypersparse matrix
%   I = sort (randperm (n, 4)) ;
%   J = sort (randperm (n, 4)) ;
%   A = magic (4) ;
%   H (I,J) = A
%   [C, I, J] = GhB.compact (H)
%   isequal (C, A)                  % C and A are the same
%   isequal (C, H(I,J))             % C and H(I,J) are the same
%   H (I, J(1)) = 0
%   [C, I, J] = GhB.compact (H, 0)
%   assert (isequal (C, A (:, 2:end))) ;
%
% See also GhB.entries, GhB.nonz, GhB.prune.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (nargin)
    case 1
        [C, I, J] = gb_compact (1, A) ;
    case 2
        [C, I, J] = gb_compact (1, A, id) ;
    case 3
        [C, I, J] = gb_compact (1, A, id, symmetric) ;
end

