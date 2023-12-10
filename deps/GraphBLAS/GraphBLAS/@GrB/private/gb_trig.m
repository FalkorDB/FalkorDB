function C = gb_trig (op, G)
%GB_TRIG inverse sine, cosine, log, sqrt, ... etc
% Implements C = asin (G), C = acos (G), C = atanh (G), ... etc

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbtype (G) ;

if (~gb_contains (type, 'complex'))

    % determine if any entries are outside the domain for the real case
    noutside = 0 ;  % default if no switch cases apply
    switch (op)

        case { 'asin', 'acos', 'atanh' }

            % C is complex if any (abs (G) > 1)
            switch (type)
                case { 'int8', 'int16', 'int32', 'int64', 'single', 'double' }
                    noutside = gbnvals (gbselect (gbapply ('abs', G), '>', 1)) ;
                case { 'uint8', 'uint16', 'uint32', 'uint64' }
                    noutside = gbnvals (gbselect (G, '>', 1)) ;
            end

        case { 'log', 'log10', 'sqrt', 'log2' }

            % C is complex if any (G < 0)
            switch (type)
                case { 'int8', 'int16', 'int32', 'int64', 'single', 'double' }
                    noutside = gbnvals (gbselect (G, '<', 0)) ;
            end

        case { 'log1p' }

            % C is complex if any (G < -1)
            switch (type)
                case { 'int8', 'int16', 'int32', 'int64', 'single', 'double' }
                    noutside = gbnvals (gbselect (G, '<', -1)) ;
            end

        case { 'acosh' }

            % C is complex if any (G < 1)
            noutside = gbnvals (gbselect (G, '<', 1)) ;
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

% if G is already complex, gbapply will select a complex operator

C = gbapply (op, G) ;

