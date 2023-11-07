function C = gb_maxbyrow (op, A)
%GB_MAXBYROW max, by row
% Implements C = max (A, [ ], 2)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% C = max (A, [ ], 2) reduces each row to a scalar; C is m-by-1
C = gbvreduce (op, A) ;

% if C(i) < 0, but if A(i,:) is sparse, then assign C(i) = 0.
ctype = gbtype (C) ;

if (gb_issigned (ctype))
    % d (i) = number of entries in A(i,:); d (i) not present if A(i,:) empty
    [m, n] = gbsize (A) ;
    d = gbdegree (A, 'row') ;
    % d (i) is an explicit zero if A(i,:) has 1 to n-1 entries
    d = gbselect (d, '<', int64 (n)) ;
    zero = gbnew (0, ctype) ;
    if (gbnvals (d) == m)
        % all rows A(i,:) have between 1 and n-1 entries
        C = gbapply2 (op, C, zero) ;
    else
        d = gbapply2 (['2nd.' ctype], d, zero) ;
        % if d(i) is between 1 and n-1 and C(i) < 0 then C(i) = 0
        C = gbeadd (op, C, d) ;
    end
end

