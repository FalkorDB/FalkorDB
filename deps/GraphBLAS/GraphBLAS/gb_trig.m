function C = gb_trig (ghb, op, G)
%GB_TRIG inverse sine, cosine, log, sqrt, ... etc.  Not user-callable.
% Implements C = asin (G), C = acos (G), C = atanh (G), ... etc

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

type = gbmex_type (G) ;

if (~gb_contains (type, 'complex'))

    % determine if any entries are outside the domain for the real case
    noutside = 0 ;  % default if no switch cases apply
    switch (op)

        case { 'asin', 'acos', 'atanh' }

            % C is complex if any (abs (G) > 1)
            switch (type)
                case { 'int8', 'int16', 'int32', 'int64', 'single', 'double' }
                    T = gzb_apply (1, 'abs', G) ;
                    noutside = gbmex_nvals (gzb_select (1, T, '>', 1)) ;
                    clear T
                case { 'uint8', 'uint16', 'uint32', 'uint64' }
                    noutside = gbmex_nvals (gzb_select (1, G, '>', 1)) ;
            end

        case { 'log', 'log10', 'sqrt', 'log2' }

            % C is complex if any (G < 0)
            switch (type)
                case { 'int8', 'int16', 'int32', 'int64', 'single', 'double' }
                    noutside = gbmex_nvals (gzb_select (1, G, '<', 0)) ;
            end

        case { 'log1p' }

            % C is complex if any (G < -1)
            switch (type)
                case { 'int8', 'int16', 'int32', 'int64', 'single', 'double' }
                    noutside = gbmex_nvals (gzb_select (1, G, '<', -1)) ;
            end

        case { 'acosh' }

            % C is complex if any (G < 1)
            noutside = gbmex_nvals (gzb_select (1, G, '<', 1)) ;
    end

    if (noutside > 0)
        % G is real but C is complex
        if (isequal (type, 'single'))
            op = [op '.single complex'] ;
        else
            op = [op '.double complex'] ;
        end
    elseif (~gb_isfloat (type))
        % G is integer or logical; use the op.double operator
        op = [op '.double'] ;
    end
end

% if G is already complex, gzb_apply will select a complex operator

C = gzb_apply (ghb, op, G) ;

