function gbtest96
%GBTEST96 test GrB.optype

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('Table of types of binary operators when inputs types are\n') ;
fprintf ('are mixed (the type of C for C=A+B, for example).\n') ;
fprintf ('\nLegend:\n') ;

types = gbtest_types ;
order = [ 3:11 2 1 12:13 ] ;
for k = order
    type = types {k} ;
    fprintf ('%2s : %s\n', gbterse (type), type) ;
end

fprintf ('\n   : ') ;
for k2 = order
    btype = types {k2} ;
    fprintf ('%2s ', gbterse (btype)) ;
end
fprintf ('\n') ;

fprintf ('---:---------------------------------------\n') ;

for k1 = order
    atype = types {k1} ;
    fprintf ('%2s : ', gbterse (atype)) ;
    for k2 = order
        btype = types {k2} ;
        ctype = GrB.optype (atype, btype) ;
        fprintf ('%2s ', gbterse (ctype)) ;

        c2 = GrB.optype (btype, atype) ;
        assert (isequal (c2, ctype)) ;

        A = GrB (1, atype) ;
        B = GrB (1, btype) ;
        c2 = GrB.optype (A, B) ;
        assert (isequal (c2, ctype)) ;

        if (isequal (atype, 'single complex'))
            A = complex (single (1)) ;
        elseif (isequal (atype, 'double complex'))
            A = complex (double (1)) ;
        else
            A = cast (1, atype) ;
        end

        c2 = GrB.optype (A, B) ;
        assert (isequal (c2, ctype)) ;

        if (isequal (btype, 'single complex'))
            B = complex (single (1)) ;
        elseif (isequal (btype, 'double complex'))
            B = complex (double (1)) ;
        else
            B = cast (1, btype) ;
        end

        c2 = GrB.optype (A, B) ;
        assert (isequal (c2, ctype)) ;

    end
    fprintf ('\n') ;
end

function s = gbterse(type)
switch (type)
    case { 'double' }
        s = 'd ' ;
    case { 'single' }
        s = 's ' ;
    case { 'logical' }
        s = 'b ' ;
    case { 'int8' }
        s = 'i1' ;
    case { 'int16' }
        s = 'i2' ;
    case { 'int32' }
        s = 'i4' ;
    case { 'int64' }
        s = 'i8' ;
    case { 'uint8' }
        s = 'u1' ;
    case { 'uint16' }
        s = 'u2' ;
    case { 'uint32' }
        s = 'u4' ;
    case { 'uint64' }
        s = 'u8' ;
    case { 'single complex' }
        s = 'c ' ;
    case { 'double complex' }
        s = 'z ' ;
    otherwise
        error ('invalid type') ;
end

