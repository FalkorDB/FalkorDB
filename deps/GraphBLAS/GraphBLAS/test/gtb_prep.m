function gtb_name = gtb_prep (ghb)
%GTB_PREP initializations for gbtests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
switch (ghb)
    case 0
        gtb_name = 'GrB' ;
    case 1
        gtb_name = 'GhB' ;
    otherwise
        gtb_name = '(random GrB/GhB)' ;
end

