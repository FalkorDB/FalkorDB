function gbtest4 (ghb)
%GBTEST4 list all semirings
% This count excludes operator synonyms ('1st' and 'first', for example),
% but it does include identical semirings with operators of different
% names.  For example, the spec has many boolean operators with different
% names but they compute the same thing.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types = gbtest_types ;
ops = gbtest_binops ;

nsemirings = 0 ;

for k1 = 1:length (ops)
    add = ops {k1} ;
    for k2 = 1:length (ops)
        mult = ops {k2} ;

        s = [add '.' mult] ;
        fprintf ('\n================================ %s\n', s) ;

        for k3 = 1:length (types)
            type = types {k3} ;

            semiring = [s '.' type] ;

            try
                gtb_semiringinfo (ghb, semiring) ;
                gtb_semiringinfo (ghb, s, type) ;
                nsemirings = nsemirings + 1 ;
                ok = true ;
            catch
                % this is an error, but it is expected since not all
                % combinations operators and types can be used to construct
                % a valid semiring.
                ok = false ;
            end
%           if (ok)
%               fprintf ('\nOK %s.%s\n', s, type) ;
%           end
        end
    end
end

fprintf ('\n') ;
gtb_semiringinfo (ghb)

fprintf ('number of semirings: %d\n', nsemirings) ;
assert (nsemirings == 2589) ;

fprintf ('\ngbtest4 (%d): all tests passed\n', ghb) ;

