function C = ktruss (A, k, symmetric)
%GHB.KTRUSS find the k-truss of a graph.
% The ktruss C is a graph consisting of a subset of the edges of A.  Each edge
% in C is part of at least k-2 triangles in A, where a triangle is a set of 3
% unique nodes that form a clique.  The pattern of C is the k-truss of A, and
% the edge weights of C are the support of each edge.   That is, C(i,j) = nt if
% the edge (i,j) is part of nt triangles in C.  All edges in C have a support
% of at least nt >= k-2.  If k=3, the total number of triangles in A is
% sum(C,'all')/6.  C is returned as a symmetric matrix with a zero-free
% diagonal.  If k defaults to 3 if not present.
%
% A must be square.  Its values are ignored; the result depends only on the
% pattern of A.  The ktruss of a matrix is only defined if it is symmetric with
% no entries on the diagonal.  Thus, C = GhB.ktruss (A, k) finds the ktruss of
% spones(A)+spones(A') after removing any diagonal entries.  If A is already
% known to have a symmetric pattern with no diagonal entries, the preprocessing
% can be skipped by using C = GhB.ktruss (A, k, 'symmetric'); in this case,
% results are undefined if A does not have these properties.
%
% To compute a sequence of k-trusses, a k1-truss can be efficiently used to
% construct another k2-truss with k2 > k1.  See the example below.
%
% The output C is symmetric with no diagonal entries, so if it is passed
% to GhB.ktruss again, the 'symmetric' option can be safely used.
%
% Example:
%
%   load west0479 ;
%   A = west0479 ;                      % A is unsymmetric
%   C3 = GhB.ktruss (A, 3) ;            % 3-truss of A+A'
%   ntriangles = sum (C3, 'all') / 6
%   C4a = GhB.ktruss (A, 4) ;           % 4-truss of A+A'
%   C4b = GhB.ktruss (C3, 4) ;          % 4-truss of A+A', just faster
%   isequal (C4a, C4b)
%
% See also GhB.tricount.

% NOTE: in GraphBLAS 10.3.0 and earlier, the optional 3rd argument was 'check',
% and the input matrix was not symmetrized.  This option has been replaced with
% 'symmetric'.  The default is now to symmetrize the matrix, which can be
% skipped if the 'symmetric' is passed in.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% get inputs
if (nargin < 2)
    k = 3 ;
end
if (k < 3)
    error ('GrB:error', 'k-truss defined only for k >= 3') ;
end

[m, n] = size (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end

% determine the types and operators to use
int_type = 'int64' ;
if (n < intmax ('int32'))
    int_type = 'int32' ;
end
semiring = ['+.oneb.' int_type ] ;
one = ['1.' int_type] ;
plus = ['oneb.' int_type] ;

% C = int32 (spones (A)) or C = int64 (spones (A))
C = GhB.apply (one, A) ;

% default: symmetrize the input matrix
symmetrize = true ;
if (nargin == 3)
    % C = GhB.ktruss (A, k, 'symmetric') ;
    % skip the preprocessing step
    symmetrize = ~isequal (symmetric, 'symmetric') ;
end
if (symmetrize)
    % remove diagonal entries from C
    GhB.select (C, C, 'offdiag', 0) ;
    % C = spones (C+C') using the oneb operator
    desc1.in1 = 'transpose' ;
    GhB.eadd (C, C, plus, C, desc1) ;
end

last_nvals = GhB.nvals (C) ;

if (GhB.isbycol (C))
    desc.in0 = 'transpose' ;
else
    desc.in1 = 'transpose' ;
end
desc.out = 'replace' ;
desc.mask = 'structural' ;

while (1)
    % C<C> = C'*C or C*C' using the plus-one semiring
    GhB.mxm (C, C, semiring, C, C, desc) ;
    % drop any entries < k-2
    GhB.select (C, C, '>=', k-2) ;
    nvals = GhB.nvals (C) ;
    if (last_nvals == nvals)
        % quit when the matrix does not change
        break ;
    end
    last_nvals = nvals ;
end

