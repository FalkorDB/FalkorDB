function result = nonz (A, varargin)
%GRB.NONZ count or query the nonzeros of a matrix.
% A GraphBLAS matrix can include explicit entries that have the value
% zero.  These entries never appear in a built-in sparse matrix.  This
% function counts or queries the nonzeros of matrix, checking their value
% and treating explicit zeros the same as entries that do not appear in
% the pattern of A.
%
% e = GrB.nonz (A)         number of nonzeros
% e = GrB.nonz (A, 'all')  number of nonzeros
% e = GrB.nonz (A, 'row')  number of rows with at least one nonzero
% e = GrB.nonz (A, 'col')  number of columns with at least one nonzero
%
% X = GrB.nonz (A, 'list')         list of values of unique nonzeros
% X = GrB.nonz (A, 'all', 'list')  list of values of unique nonzeros
% I = GrB.nonz (A, 'row', 'list')  list of rows with at least one nonzero
% J = GrB.nonz (A, 'col', 'list')  list of cols with at least one nonzero
%
% d = GrB.nonz (A, 'row', 'degree')
%   If A is m-by-n, then d is a sparse column vector of size m, with d(i)
%   equal to the number of nonzeros in A(i,:).  If A(i,:) has no
%   nonzeros, then d(i) is an implicit zero, not present in the pattern
%   of d, so I = find (d) is the same I = GrB.nonz (A, 'row', 'list').
%
% d = GrB.nonz (A, 'col', 'degree')
%   If A is m-by-n, d is a sparse column vector of size n, with d(j)
%   equal to the number of nonzeros in A(:,j).  If A(:,j) has no
%   nonzeros, then d(j) is an implicit zero, not present in the pattern
%   of d, so I = find (d) is the same I = GrB.nonz (A, 'col', 'list').
%
% With an optional scalar argument as the last argument, the value of the
% 'zero' can be specified; d = GrB.nonz (A, ..., id).  For example, to
% count all entries in A not equal to one, use GrB.nonz (A, 1).
%
% The result is a built-in scalar or vector, except for the 'degree'
% usage, in which case the result is a GrB vector d.
%
% Example:
%
%   A = magic (5) ;
%   A (A < 10) = 0              % built-in full matrix with explicit zeros
%   nnz (A)
%   GrB.nonz (A)                % same as nnz (A)
%   G = GrB (A)                 % contains explicit zeros
%   GrB.nonz (G)                % same as nnz (A)
%   G (A > 18) = sparse (0)     % entries A>18 deleted, explicit zeros
%   GrB.nonz (G)
%   GrB.nonz (G, 'list')
%   S = double (G)              % built-in sparse matrix; no explicit zeros
%   GrB.nonz (S)
%   GrB.nonz (S, 'list')
%
% See also GrB.entries, GrB/nnz, GrB/nonzeros, GrB.prune.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

builtin_sparse = false ;
if (isobject (A))
    % A is a GraphBLAS matrix; get its opaque content
    A = A.opaque ;
elseif (builtin ('issparse', A))
    % A is a built-in sparse matrix
    builtin_sparse = true ;
end

% get the identity value
id = 0 ;
nargs = nargin ;
if (nargin > 1)
    lastarg = varargin {nargs-1} ;
    if (~ischar (lastarg))
        % the last argument is id, if it is not a string
        id = gb_get_scalar (lastarg) ;
        nargs = nargs - 1 ;
    end
end

if (id ~= 0)
    % id is nonzero, so prune A first (for any matrix A)
    A = gbselect (A, '~=', id) ;
elseif (~builtin_sparse)
    % id is zero, so prune A only if it is a GraphBLAS matrix,
    % or a built-in full matrix.  A built-in sparse matrix can remain
    % unchanged.
    A = gbselect (A, 'nonzero') ;
end

% get the count/list of the entries of A
result = gb_entries (A, varargin {1:nargs-1}) ;

% if gb_entries returned a GraphBLAS struct, return it as a GrB matrix
if (isstruct (result))
    result = GrB (result) ;
end

