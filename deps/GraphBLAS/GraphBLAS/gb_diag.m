function C = gb_diag (ghb, A, k)
%GB_DIAG implements GrB/diag and GhB/diag.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin < 3)
    k = 0 ;
end
if (isobject (k))
    k = gb_get_scalar (k) ;
end

[am, an] = gbmex_size (A) ;

if (am == 1)

    % C = diag (v,k) where A is a row vector and C is a matrix
    C = gzb_mdiag (ghb, gzb_trans (1, A), k) ;

elseif (an == 1)

    % C = diag (v,k) where A is a column vector and C is a matrix
    C = gzb_mdiag (ghb, A, k) ;

else

    % v = diag (A,k) is a column vector formed from the elements of the
    % kth diagonal of A
    C = gzb_vdiag (ghb, A, k) ;

end

