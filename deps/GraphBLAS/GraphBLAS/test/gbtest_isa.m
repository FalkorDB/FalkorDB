function s = gbtest_isa (ghb, G)
%GBTEST_ISA check if G has the GrB or GrB "isa" property.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (ghb)
    case 0
        s = (isa (G, 'GrB')) ;
    case 1
        s = (isa (G, 'GhB')) ;
    otherwise
        s = (isa (G, 'GhB') || isa (G, 'GrB')) ;
end

