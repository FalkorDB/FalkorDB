function C = gb_maxbyrow (ghb, op, A)
%GB_MAXBYROW max, by row.  Not user-callable.
% Implements C = max (A, [ ], 2)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% C = max (A, [ ], 2) reduces each row to a scalar; C is m-by-1
C = gzb_vreduce (ghb, A, op) ;

% if C(i) < 0, but if A(i,:) is sparse, then assign C(i) = 0.
ctype = gb_type (C) ;

if (gb_issigned (ctype))
    % d (i) = number of entries in A(i,:); d (i) not present if A(i,:) empty
    [m, n] = gbmex_size (A) ;
    d = gzb_degree (1, A, 'row') ;
    % d (i) is an explicit zero if A(i,:) has 1 to n-1 entries
    s = gzb_select (1, d, '<', int64 (n)) ;
    zero = gzb (1, 0, ctype) ;
    if (gb_is_grb (C))
        C = struct (C) ;
    end
    if (gbmex_nvals (s) == m)
        % all rows A(i,:) have between 1 and n-1 entries
        C = gzb_apply2 (ghb, op, C, zero) ;
    else
        z = gzb_apply2 (1, ['2nd.' ctype], s, zero) ;
        % if z(i) is between 1 and n-1 and C(i) < 0 then C(i) = 0
        C = gzb_eadd (ghb, op, C, z) ;
    end
end

