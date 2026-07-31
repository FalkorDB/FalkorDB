function C = incidence (A, varargin)
%GRB.INCIDENCE graph incidence matrix.
% C = GrB.incidence (A) is the graph incidence matrix of the square matrix A.
% C is GraphBLAS matrix of size n-by-e, if A is n-by-n with e entries (not
% including diagonal entries).  The jth column of C has 2 entries: C(s,j) = -1
% and C(t,j) = 1, where A(s,t) is an entry A.  Diagonal entries in A are
% ignored.
%
%   C = GrB.incidence (A, ..., 'directed') constructs a matrix C of size n-by-e
%       where e = GrB.entries (GrB.offdiag (A)).  Any entry in the upper or
%       lower trianglar part of A results in a unique column of C.  The
%       diagonal is ignored.  This is the default.
%
%   C = GrB.incidence (A, ..., 'unsymmetric') is the same as 'directed'.
%
%   C = GrB.incidence (A, ..., 'undirected') assumes A is symmetric, and only
%       creates columns of C based on entries in tril (A,-1).  The diagonal and
%       upper triangular part of A are ignored.
%
%   C = GrB.incidence (A, ..., 'symmetric') is the same as 'undirected'.
%
%   C = GrB.incidence (A, ..., 'lower') is the same as 'undirected'.
%
%   C = GrB.incidence (A, ..., 'upper') is the same as 'undirected',
%       except that only entries in triu (A,1) are used.
%
%   C = GrB.incidence (A, ..., type) constructs C with the type 'double',
%       'single', 'int8', 'int16', 'int32', or 'int64'.  The default is
%       'double'.  Unsigned types are not allowed C must contain -1's.
%
% Examples:
%
%   A = sprand (5, 5, 0.5)
%   C = GrB.incidence (A)
%
% See also graph/incidence, digraph/incidence.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_incidence (0, A, varargin {:}) ;

