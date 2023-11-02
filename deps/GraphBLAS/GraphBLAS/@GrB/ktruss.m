function C = ktruss (A, k, check)
%GRB.KTRUSS find the k-truss of a matrix.
% C = GrB.ktruss (A, k) finds the k-truss of a matrix A.  spones (A) must
% be symmetric with no diagonal entries.  Only the pattern of A is
% considered.  The ktruss C is a graph consisting of a subset of the edges
% of A.  Each edge in C is part of at least k-2 triangles in A, where a
% triangle is a set of 3 unique nodes that form a clique.  The pattern of C
% is the k-truss, and the edge weights of C are the support of each edge.
% That is, C(i,j) = nt if the edge (i,j) is part of nt triangles in C.  All
% edges in C have a support of at least nt >= k-2.  The total number of
% triangles in C is sum(C,'all')/6.  C is returned as a symmetric matrix
% with a zero-free diagonal.  If k is not present, it defaults to 3.
%
% To compute a sequence of k-trusses, a k1-truss can be efficiently used to
% construct another k2-truss with k2 > k1.
%
% To check the input A to make sure it has a symmetric pattern and has a
% zero-free diagonal, use C = GrB.ktruss (A, k, 'check').  This check is
% optional since it adds extra time.  Results are undefined if 'check' is
% not specified and A has an unsymmetric pattern or entries on the
% diagonal.
%
% The output C is symmetric with a zero-free diagonal.
%
% Example:
%
%   load west0479 ;
%   A = GrB.offdiag (west0479) ;
%   A = A+A' ;
%   C3 = GrB.ktruss (A, 3) ;
%   ntriangles = sum (C3, 'all') / 6
%   C4a = GrB.ktruss (A, 4) ;
%   C4b = GrB.ktruss (C3, 4) ;          % this is faster
%   isequal (C4a, C4b)
%
% See also GrB.tricount.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% NOTE: this is a high-level algorithm that uses GrB objects.

% check inputs
if (nargin < 2)
    k = 3 ;
end
if (k < 3)
    error ('GrB:error', 'k-truss defined only for k >= 3') ;
end

if (nargin < 3)
    check = false ;
else
    check = isequal (check, 'check') ;
end

[m, n] = size (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end

int_type = 'int64' ;
if (n < intmax ('int32'))
    int_type = 'int32' ;
end

C = GrB.apply (['1.' int_type], A) ;

if (check)
    % Do the costly checks.  These are optional.
    if (~issymmetric (C))
        error ('GrB:error', 'A must have a symmetric pattern') ;
    end
    if (nnz (diag (C) > 0))
        error ('GrB:error', 'A must have a zero-free diagonal') ;
    end
end

lastnz = nnz (C) ;

while (1)
    % C<C> = C*C using the plus-and semiring, then drop any < k-2.
    C = GrB.select (GrB.mxm (C, C, '+.&', C, C), '>=', k-2) ;
    nz = nnz (C) ;
    if (lastnz == nz)
        % quit when the matrix does not change
        break ;
    end
    lastnz = nz ;
end

