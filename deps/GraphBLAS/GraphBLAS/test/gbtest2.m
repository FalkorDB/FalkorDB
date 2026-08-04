function gbtest2 (ghb)
%GBTEST2 list all binary operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

optype = gbtest_types ;
opnames = gbtest_binops ;
nbinop = 0 ;

for k1 = 1:length(opnames)

    opname = opnames {k1} ;
    fprintf ('\n=================================== %s\n', opname) ;

    for k2 = 0:length(optype)

        op = opname ;
        if (k2 > 0)
            op = [op '.' optype{k2}] ; %#ok<*AGROW>
        end

        % fprintf ('\nop: (%s)\n', op) ;
        try
            if (k2 > 0)
                gtb_binopinfo (ghb, op) ;
                nbinop = nbinop + 1 ;
            else
                gtb_binopinfo (ghb, op, 'double') ;
            end
        catch
        end
    end
end

fprintf ('\nhelp GrB.binopinfo:\n') ;
gtb_binopinfo (ghb) ;

fprintf ('number of valid binary operators: %d\n', nbinop) ;
assert (nbinop == 414) ;

fprintf ('gbtest2 (%d): all tests passed\n', ghb) ;

