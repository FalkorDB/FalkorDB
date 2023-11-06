function C = incidence (A, varargin)
%GRB.INCIDENCE graph incidence matrix.
% C = GrB.incidence (A) is the graph incidence matrix of the square
% matrix A.  C is GraphBLAS matrix of size n-by-e, if A is n-by-n with e
% entries (not including diagonal entries).  The jth column of C has 2
% entries: C(s,j) = -1 and C(t,j) = 1, where A(s,t) is an entry A.
% Diagonal entries in A are ignored.
%
%   C = GrB.incidence (A, ..., 'directed') constructs a matrix C of size
%       n-by-e where e = GrB.entries (GrB.offdiag (A)).  Any entry in the
%       upper or lower trianglar part of A results in a unique column of
%       C.  The diagonal is ignored.  This is the default.
%
%   C = GrB.incidence (A, ..., 'unsymmetric') is the same as 'directed'.
%
%   C = GrB.incidence (A, ..., 'undirected') assumes A is symmetric, and
%       only creates columns of C based on entries in tril (A,-1).  The
%       diagonal and upper triangular part of A are ignored.
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
%       'double'.  The type cannot be 'logical' or 'uint*' since C
%       must contain -1's.
%
% Examples:
%
%   A = sprand (5, 5, 0.5)
%   C = GrB.incidence (A)
%
% See also graph/incidence, digraph/incidence.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

[m, n] = gbsize (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end

% get the string options
kind = 'directed' ;
type = 'double' ;
for k = 1:nargin-1
    arg = lower (varargin {k}) ;
    switch arg
        case { 'directed', 'undirected', 'symmetric', 'unsymmetric', ...
            'lower', 'upper' }
            kind = arg ;
        case { 'double', 'single', 'int8', 'int16', 'int32', 'int64' }
            type = arg ;
        case { 'uint8', 'uint16', 'uint32', 'uint64', 'logical' }
            error ('GrB:error', 'type must be signed') ;
        otherwise
            error ('GrB:error', 'unknown option') ;
    end
end

switch (kind)

    case { 'directed', 'unsymmetric' }

        % create the incidence matrix of a directed graph, using all of A;
        % except that diagonal entries are ignored.
        A = gbselect ('offdiag', A, 0) ;

    case { 'upper' }

        % create the incidence matrix of an undirected graph, using only
        % entries in the strictly upper triangular part of A.
        A = gbselect ('triu', A, 1) ;

    otherwise   % 'undirected', 'symmetric', or 'lower'

        % create the incidence matrix of an undirected graph, using only
        % entries in the strictly lower triangular part of A.
        A = gbselect ('tril', A, -1) ;

end

% build the incidence matrix
desc.base = 'zero-based' ;
[I, J] = gbextracttuples (A, desc) ;
e = length (I) ;
I = [I ; J] ;
J = (int64 (0) : int64 (e-1))' ;
J = [J ; J] ;
X = ones (e, 1, type) ;
X = [-X ; X] ;
C = GrB (gbbuild (I, J, X, n, e, desc)) ;

