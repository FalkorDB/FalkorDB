function L = laplacian (A, type, check)
%GRB.LAPLACIAN Laplacian matrix
% L = laplacian (A) is the graph Laplacian of the matrix A.  spones(A)
% must be symmetric.  The diagonal of A is ignored. The diagonal of L is
% the degree of the nodes.  That is, L(j,j) = sum (spones (A (:,j))),
% assuming A has no diagonal entries..  For off-diagonal entries, L(i,j) =
% L(j,i) = -1 if the edge (i,j) exists in A.
%
% The type of L defaults to double.  With a second argument, the type of L
% can be specified, as L = laplacian (A,type); type may be 'double',
% 'single', 'int8', 'int16', 'int32', 'int64', 'single complex', or
% 'double complex'.  Be aware that integer overflow may occur with the
% smaller integer types, if the degree of any nodes exceeds the largest
% integer value.
%
% spones(A) must be symmetric on input, but this condition is not checked
% by default.  If it is not symmetric, the results are undefined.  To
% check this condition, use GrB.laplacian (A, 'double', 'check') ;
%
% L is returned as symmetric GraphBLAS matrix.
%
% Example:
%
%   A = bucky ;
%   L = GrB.laplacian (A)
%
% See also graph/laplacian.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

[m, n] = gbsize (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square and symmetric') ;
end

% get the type
if (nargin < 2)
    type = 'double' ;
elseif (~gb_issigned (type))
    % type must be signed
    error ('GrB:error', 'type cannot be logical or unsigned integer') ;
end

% S = spones (A)
S = gbapply (['1.' type], A) ;

% check the input matrix, if requested
if (nargin > 2 && isequal (check, 'check'))
    % make sure spones (S) is symmetric
    if (~gb_issymmetric (S, 'nonskew', false))
        error ('GrB:error', 'spones(A) must be symmetric') ;
    end
end

% D = diagonal matrix with d(i,i) = row/column degree of node i
fmt = gbformat (S) ;
if (isequal (fmt, 'by row'))
    D = gbdegree (S, 'row') ;
else
    D = gbdegree (S, 'col') ;
end
D = gbmdiag (D, 0) ;
if (~isequal (type, gbtype (D)))
    % gbdegree returns its result as int64; typecast to desired type
    D = gbnew (D, type) ;
end

% construct the Laplacian
% L = D-S
L = GrB (gbeadd (D, '+', gbapply ('-', S))) ;

