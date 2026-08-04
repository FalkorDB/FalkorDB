function C = gzb_cat (ghb, Tiles)
%GZB_CAT: wrapper for gbmex_cat mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

n = numel (Tiles) ;
for k = 1:n
    if (gb_is_grb (Tiles {k}))
        Tiles {k} = struct (Tiles {k}) ;
    end
end

if (ghb)
    C = GhB (gbmex_cat (1, Tiles)) ;
else
    C = GrB (gbmex_cat (0, Tiles)) ;
end

