function C = offdiag (A)
%GHB.OFFDIAG remove diaogonal entries.
% C = GhB.offdiag (A) removes diagonal entries from A.
%
% See also GhB/tril, GhB/triu, GhB/diag, GhB.select.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gzb_select (1, 'offdiag', A, 0) ;

