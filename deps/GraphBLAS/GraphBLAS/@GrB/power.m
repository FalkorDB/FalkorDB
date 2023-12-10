function C = power (A, B)
%.^ array power.
% C = A.^B computes element-wise powers.  One or both of A and B may be
% scalars.  Otherwise, A and B must have the same size.  C is sparse (with
% the same pattern as A) if B is a positive scalar (greater than zero), or
% full otherwise.
%
% See also GrB/mpower, GrB/pow2, GrB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end
if (isobject (B))
    B = B.opaque ;
end

C = GrB (gb_power (A, B)) ;

