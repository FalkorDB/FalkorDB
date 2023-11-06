function C = max (A, B, option)
%MAX Maximum elements of a matrix.
% C = max (A) is the largest entry in the vector A.  If A is a matrix,
% C is a row vector with C(j) = max (A (:,j)).
%
% C = max (A,B) is an array of the element-wise maximum of two matrices
% A and B, which either have the same size, or one can be a scalar.
%
% C = max (A, [ ], 'all') is a scalar, with the largest entry in A.
% C = max (A, [ ], 1) is a row vector with C(j) = max (A (:,j))
% C = max (A, [ ], 2) is a column vector with C(i) = max (A (i,:))
%
% The 2nd output of [C,I] = max (...) in the built-in max
% is not supported; see GrB.argmax instead.  The max (..., nanflag)
% not yet supported; only the 'omitnan' behavior is supported.
%
% Complex matrices are not supported.
%
% See also GrB/min, GrB.argmax.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

type = gbtype (A) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'complex matrices not yet supported') ;
elseif (isequal (type, 'logical'))
    op = '|.logical' ;
else
    op = 'max' ;
end

if (nargin == 1)
    % C = max (A)
    C = GrB (gb_max1 (op, A)) ;
elseif (nargin == 2)
    % C = max (A,B)
    if (isobject (B))
        B = B.opaque ;
    end
    C = GrB (gb_max2 (op, A, B)) ;
else
    % C = max (A, [ ], option)
    if (~isempty (B))
        error ('GrB:error', ...
            'dimension argument not allowed with 2 input matrices') ;
    end
    C = GrB (gb_max3 (op, A, option)) ;
end

