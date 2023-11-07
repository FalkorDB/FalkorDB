function gbtest4
%GBTEST4 list all semirings
% This count excludes operator synonyms ('1st' and 'first', for example),
% but it does include identical semirings with operators of different
% names.  For example, the spec has many boolean operators with different
% names but they compute the same thing.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

types = gbtest_types ;
ops = gbtest_binops ;

nsemirings = 0 ;

for k1 = 1:length (ops)
    add = ops {k1} ;
    for k2 = 1:length (ops)
        mult = ops {k2} ;
        for k3 = 1:length (types)
            type = types {k3} ;

            s = [add '.' mult] ;
            semiring = [s '.' type] ;

            fprintf ('\n================================ %s\n', semiring) ;

            try
                GrB.semiringinfo (semiring) ;
                GrB.semiringinfo (s, type) ;
                nsemirings = nsemirings + 1 ;
            catch
                % this is an error, but it is expected since not all
                % combinations operators and types can be used to construct
                % a valid semiring.
            end
        end
    end
end

fprintf ('\n') ;
GrB.semiringinfo

fprintf ('number of semirings: %d\n', nsemirings) ;
assert (nsemirings == 2589) ;

fprintf ('\ngbtest4: all tests passed\n') ;

