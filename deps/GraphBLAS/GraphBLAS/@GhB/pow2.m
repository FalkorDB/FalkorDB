function C = pow2 (A, B)
%POW2 base-2 power and scale floating-point number.
% C = pow2 (A) is C(i,j) = 2.^A(i,j) for each entry in A.  Since 2^0 is
% nonzero, C is a full matrix.
%
% C = pow2 (F,E) is C = F .* (2 .^ fix (E)).  C is sparse, with the same
% pattern as F+E.  Any imaginary parts of F and E are ignored.
%
% See also GhB/log2, GhB/power, GhB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    C = gb_pow2 (1, A) ;
else
    C = gb_pow2 (1, A, B) ;
end

