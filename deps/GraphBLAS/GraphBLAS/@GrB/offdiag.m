function C = offdiag (A)
%GRB.OFFDIAG remove diaogonal entries.
% C = GrB.offdiag (A) removes diagonal entries from A.
%
% See also GrB/tril, GrB/triu, GrB/diag, GrB.select.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

C = GrB (gbselect ('offdiag', A, 0)) ;

