function C = build (varargin)
%GRB.BUILD construct a sparse matrix from a list of entries.
%
%   C = GrB.build (I, J, X, m, n, dup, type, desc)
%
% GrB.build constructs an m-by-n GraphBLAS sparse matrix C from a list of
% entries, analogous to A = sparse (I, J, X, m, n) to construct a
% sparse matrix A.
%
% If not present or empty, m defaults to the largest row index in the
% list I, and n defaults to the largest column index in the list J.  dup
% defaults to '+' for non-logical types, and 'or' for logical, which gives
% the same behavior as the built-in sparse function: duplicate entries are
% added together.
%
% dup is a string that defines a binary function; see 'help GrB.binopinfo'
% for a list of available binary operators.  The dup operator need not be
% associative.  If two entries in [I,J,X] have the same row and column
% index, the dup operator is applied to assemble them into a single
% entry.  Suppose (i,j,x1), (i,j,x2), and (i,j,x3) appear in that order
% in [I,J,X], in any location (the arrays [I J] need not be sorted, and
% so these entries need not be adjacent).  That is, i = I(k1) = I(k2) =
% I(k3) and j = J(k1) = J(k2) = J(k3) for some k1 < k2 < k3.  Then C(i,j)
% is computed as follows, in order:
%
%   x = X (k1) ;
%   x = dup (x, X (k2)) ;
%   x = dup (x, X (k3)) ;
%   C (i,j) = x ;
%
% For example, if the dup operator is '1st', then C(i,j)=X(k1) is set,
% and the subsequent entries are ignored.  If dup is '2nd' or 'ignore'
% then C(i,j)=X(k3), and the preceding entries are ignored.  If dup
% is the empty string ('') then duplicates result in an error.
%
% type is a string that defines the type of C (see 'help GrB' for a list
% of types).  If the type is not specified, it defaults to the type of X.
%
% The integer arrays I and J may be double, int64, or uint64:
% If I, J, and X are double, the following examples construct the same
% sparse matrix S:
%
%   S = sparse (I, J, X) ;
%   S = GrB.build (I, J, X, struct ('kind', 'sparse')) ;
%   S = double (GrB.build (I, J, X)) ;
%   S = double (GrB.build (uint64(I), uint64(J), X)) ;
%
% The row and column indices I and J need not be in any particular order,
% but GrB.build is fastest if I and J are provided in column-major order
% if building a built-in sparse matrix.  If desc.format is 'by row', then
% GrB.build is fastest if I and J are in row-major order.
%
% If desc.base is 'zero-based', and I and J are int64 or uint64, then I
% and J are treated as zero-based, where (0,0) is the first entry in the
% top left of S, and (m-1,n-1) is the position in the bottom right corner
% of S.  GrB.build is fastest if I and J are int64 or uint64, and
% desc.base is 'zero-based'.  The default is the same as built-in
% indexing, which is 'one-based'.  desc.base is ignored if I and J are
% double.
%
% If I, J, and/or X are scalars, and any of I, J, or X is a vector of
% length e, the scalars are expanded into vectors of length e.  Any
% vectors I, J, and X must all be the same size, e.
%
% To build an iso-valued matrix, X must a scalar, and the dup operator
% must be one of: '1st', '2nd', 'any', 'min', 'max', 'or', 'and',
% 'bitor', 'bitand', or dup may be omitted if X is logical.
%
% See also sparse, GrB/sparse, GrB/find, GrB.extracttuples.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[C, k] = gbbuild (varargin {:}) ;
if (k == 0)
    C = GrB (C) ;
end

