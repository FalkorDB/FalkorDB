function binops = gb_binops
%GB_BINOPS: return a list of all binary ops and their descriptions

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

binops = {
    '1st'       ,   '1st(x,y) = x' ;
    '2nd'       ,   '2nd(x,y) = y' ;
    'oneb'      ,   'oneb(x,y) = 1' ;
    'pair'      ,   'pair(x,y) = 1' ;
    'any'       ,   'any(x,y) is x or y, chosen arbitrarily' ;
    'min'       ,   '' ;
    'max'       ,   '' ;
    '+'         ,   '' ;
    '-'         ,   '' ;
    'rminus'    ,   'rminus(x,y) is y-x' ;
    '*'         ,   '' ;
    '/'         ,   '' ;
    '\'         ,   '\ is reverse division, y/x' ;
    'iseq'      ,   'iseq(x,y) = (x==y) = 0 or 1 with the given type' ;
    'isne'      ,   'isne(x,y) = (x~=y) = 0 or 1 with the given type' ;
    'isgt'      ,   'isgt(x,y) = (x>y) = 0 or 1 with the given type' ;
    'islt'      ,   'islt(x,y) = (x<y) = 0 or 1 with the given type' ;
    'isge'      ,   'isge(x,y) = (x>=y) = 0 or 1 with the given type' ;
    'isle'      ,   'isle(x,y) = (x<=y) 0 or 1 with the given type' ;
    '=='        ,   '' ;
    '~='        ,   '' ;
    '>'         ,   '' ;
    '<'         ,   '' ;
    '>='        ,   '' ;
    '<='        ,   '' ;
    '|'         ,   '' ;
    '&'         ,   '' ;
    'xor'       ,   '' ;
    'atan2'     ,   '' ;
    'hypot'     ,   '' ;
    'fmod'      ,   'fmod(x,y) = x-fix(x/y)*y' ;
    'remainder' ,   'remainder(x,y) = x-round(x/y)*y' ;
    'copysign'  ,   'copysign(x,y) = abs(x)*sign(y)' ;
    'cmplx'     ,   'cmplx(x,y) = x+i*y' ;
    'pow2'      ,   'pow2(x,y) = x*2^y' ;
    'xnor'      ,   '' ;
    'pow'       ,   'pow(x,y) = x^y' ;
    'bitor'     ,   '' ;
    'bitand'    ,   '' ;
    'bitxor'    ,   '' ;
    'bitxnor'   ,   'bitxnor(x,y) is the bit-wise xnor, ~bitxor(x,y)'
    'firsti0'   ,   'the row index of x in its matrix/vector, minus 1' ;
    'firsti1'   ,   'the row index of x in its matrix/vector' ;
    'firstj0'   ,   'the column index of x in its matrix/vector, minus 1' ;
    'firstj1'   ,   'the column index of x in its matrix/vector' ;
    'secondi0'  ,   'the row index of y in its matrix/vector, minus 1' ;
    'secondi1'  ,   'the row index of y in its matrix/vector' ;
    'secondj0'  ,   'the column index of y in its matrix/vector, minus 1' ;
    'secondj1'  ,   'the column index of y in its matrix/vector' ;
    } ;

