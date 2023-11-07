function result = entries (A, varargin)
%GRB.ENTRIES count or query the entries of a matrix.
% An entry A(i,j) in a GraphBLAS matrix is one that is present in the
% data structure.  Unlike a built-in sparse matrix, a GraphBLAS matrix can
% contain explicit zero entries.  All entries in a built-in sparse matrix
% are nonzero.  A built-in full matrix has all of its entries present,
% regardless of their value.  The GrB.entries function looks only at the
% pattern of A, not its values.  To exclude explicit entries with a value
% of zero (or any specified additive identity value) use GrB.nonz
% instead.
%
% Let [m n] = size (A)
%
% e = GrB.entries (A)         number of entries
% e = GrB.entries (A, 'all')  number of entries
% e = GrB.entries (A, 'row')  number of rows with at least one entry
% e = GrB.entries (A, 'col')  number of columns with at least one entry
%
% X = GrB.entries (A, 'list')         list of values of unique entries
% X = GrB.entries (A, 'all', 'list')  list of values of unique entries
% I = GrB.entries (A, 'row', 'list')  list of rows with at least one entry
% J = GrB.entries (A, 'col', 'list')  list of cols with at least one entry
%
% d = GrB.entries (A, 'row', 'degree')
%   If A is m-by-n, then d is a sparse column vector of size m, with d(i)
%   equal to the number of entries in A(i,:).  If A(i,:) has no entries,
%   then d(i) is an implicit zero, not present in the pattern of d, so
%   that I = find (d) is the same I = GrB.entries (A, 'row', 'list').
%
% d = GrB.entries (A, 'col', 'degree')
%   If A is m-by-n, d is a sparse column vector of size n, with d(j)
%   equal to the number of entries in A(:,j).  If A(:,j) has no entries,
%   then d(j) is an implicit zero, not present in the pattern of d, so
%   that I = find (d) is the same I = GrB.entries (A, 'col', 'list').
%
% The result is a built-in scalar or vector, except for the 'degree'
% usage, in which case the result is a GrB vector d.
%
% Example:
%
%   A = magic (5) ;
%   A (A < 10) = 0             % built-in full matrix with some explicit zeros
%   GrB.entries (A)            % all entries present in a built-in full matrix
%   G = GrB (A)                % contains explicit zeros
%   GrB.entries (G)
%   G (A > 18) = sparse (0)    % entries A>18 deleted, has explicit zeros
%   GrB.entries (G)
%   GrB.entries (G, 'list')
%   S = double (G)             % built-in sparse matrix; no explicit zeros
%   GrB.entries (S)
%   GrB.entries (S, 'list')
%
% See also GrB.nonz, nnz, GrB/nnz, nonzeros, GrB/nonzeros.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    % A is a GraphBLAS matrix; get its opaque content
    A = A.opaque ;
end

% get the count/list of the entries of A
result = gb_entries (A, varargin {:}) ;

% if gb_entries returned a GraphBLAS struct, return it as a GrB matrix
if (isstruct (result))
    result = GrB (result) ;
end

