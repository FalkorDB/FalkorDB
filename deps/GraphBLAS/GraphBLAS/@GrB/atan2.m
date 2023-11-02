function C = atan2 (A, B)
%ATAN2 four quadrant inverse tangent.
% C = atan2 (X,Y) is the 4 quadrant arctangent of the entries in X and Y.
%
% See also GrB/tan, GrB/tanh, GrB/atan, GrB/atanh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

atype = gbtype (A) ;
btype = gbtype (B) ;

if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
    error ('GrB:error', 'inputs must be real') ;
end

if (~gb_isfloat (atype))
    A = gbnew (A, 'double') ;
end

if (~gb_isfloat (btype))
    B = gbnew (B, 'double') ;
end

% atan2(A,B) gives the set union of the pattern of A and B

if (gb_isscalar (A))
    if (gb_isscalar (B))
        % both A and B are scalars
        C = GrB (gbemult ('atan2', A, B)) ;
    else
        % A is a scalar, B is a matrix
        C = GrB (gbapply2 ('atan2', gbfull (A), B)) ;
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        C = GrB (gbapply2 ('atan2', A, gbfull (B))) ;
    else
        % both A and B are matrices.  C is the set union of A and B.
        C = GrB (gbeunion ('atan2', A, 0, B, 0)) ;
    end
end

