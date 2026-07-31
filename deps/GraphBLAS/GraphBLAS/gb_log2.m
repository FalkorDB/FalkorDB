function [F, E] = gb_log2 (ghb, G)
%GB_LOG2 implements log2 for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (nargout == 1)
    % F = log2 (G)
    F = gb_trig (ghb, 'log2', gzb_full (1, G)) ;
    if (gb_make_real (F))
        F = gzb_apply (ghb, 'creal', F) ;
    end
else
    % [F,E] = log2 (G)
    type = gbmex_type (G) ;
    switch (type)
        case { 'logical', 'int8', 'int16', 'int32', 'int64', ...
            'uint8', 'uint16', 'uint32', 'uint64', 'double complex' }
            type = 'double' ;
        case { 'single complex' }
            type = 'single' ;
        case { 'single', 'double' }
            % type remains the same
    end
    F = gzb_apply (ghb, ['frexpx.' type], G) ;
    E = gzb_apply (ghb, ['frexpe.' type], G) ;
end

