function test153
%TEST153 list all possible semirings
%
% Lists all possible semirings that can be constructed from built-in operators.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, unary_ops, add_ops, types, semirings, selops] = GB_spec_opsall ;

n = 0 ;
types = types.all ;
binops = binops.all ;

for kadd = 1:length (add_ops)
    for kadd_types = 1:length (types)
        add.opname = add_ops {kadd} ;
        add.optype = types {kadd_types} ;

        try
            [opadd, t, ztype, xtype, ytype] = GB_spec_operator (add) ;
            id = GB_spec_identity (add) ;
        catch
            continue ;
        end

        if (~isequal (opadd, add.opname))
            % ignore renames
            continue ;
        end

        if (~isequal (ztype, xtype))
            continue ;
        end

        if (~isequal (ztype, ytype))
            continue ;
        end

        fprintf ('\n======================= monoid %s.%s:\n', opadd, ztype) ;

        n2 = 0 ;
        for kmult = 1:length (binops)
            for kmult_types = 1:length (types)

                mult.opname = binops {kmult} ;
                mult.optype = types {kmult_types} ;

                try
                    [opmult, t2, z2, xtype, y] = GB_spec_operator (mult) ;
                catch
                    continue ;
                end

                if (~isequal (opmult, mult.opname))
                    % ignore renames
                    continue ;
                end

                if (isequal (z2, ztype))
                    n2 = n2+1 ;
                    fprintf ('    %s.%s.%s\n', opadd, opmult, xtype) ;
                end
            end
        end

        fprintf ('    semirings with %s.%s: %d\n', opadd, ztype, n2) ;
        n = n + n2 ;
    end
end

fprintf ('total unique semirings: %d\n', n) ;

