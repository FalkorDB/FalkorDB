function C = gb_maxbycol (ghb, op, A)
%GB_MAXBYCOL max, by column.  Not user-callable.
% Implements C = max (A, [ ], 1)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% C = max (A, [ ], 1) reduces each col to a scalar; C is 1-by-n
desc.in0 = 'transpose' ;
C = gzb_vreduce (1, A, op, desc) ;

% if C(j) < 0, but if A(:,j) is sparse, then assign C(j) = 0.
ctype = gbmex_type (C) ;

if (gb_issigned (ctype))
    % d (j) = number of entries in A(:,j); d (j) not present if A(:,j) empty
    [m, n] = gbmex_size (A) ;
    d = gzb_degree (1, A, 'col') ;
    % s (j) is an explicit zero if A(:,j) has 1 to m-1 entries
    s = gzb_select (1, d, '<', int64 (m)) ;
    zero = gzb (1, 0, ctype) ;
    if (gbmex_nvals (s) == n)
        % all columns A(:,j) have between 1 and m-1 entries
        C = gzb_apply2 (1, op, C, zero) ;
    else
        z = gzb_apply2 (1, ['2nd.' ctype], s, zero) ;
        % if z (j) is between 1 and m-1 and C (j) < 0 then C (j) = 0
        C = gzb_eadd (1, op, C, z) ;
    end
end

C = gzb_trans (ghb, C) ;

