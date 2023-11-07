function done = long_enough (tm, tg, k)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (k < 3)
    done = false ;
else
    t1 = sum (tm (1:k)) ;
    t2 = sum (tg (1:k)) ;
    done = (max (t1, t2) > 3) ;
end

