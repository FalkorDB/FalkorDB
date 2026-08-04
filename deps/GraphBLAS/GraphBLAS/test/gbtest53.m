function gbtest53 (ghb)
%GBTEST53 test [GrB,GhB].monoidinfo

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types10 = {
    'double'
    'single'
    'int8'
    'int16'
    'int32'
    'int64'
    'uint8'
    'uint16'
    'uint32'
    'uint64'
    } ;

nmonoids = 0 ;

% 50 real monoids (integer and floating-point, not logical):
ops = { '+', '*', 'min', 'max', 'any' } ;
for k1 = 1:5
    op = ops {k1} ;
    fprintf ('\nop ( %s )=============================================\n', op) ;
    for k2 = 1:10
        type = types10 {k2} ;
        gtb_monoidinfo (ghb, [op '.' type]) ;
        gtb_monoidinfo (ghb, op, type) ;
        nmonoids = nmonoids + 1 ;
    end
end

% 5 boolean monoids:
ops = { '|', '&', 'xor', 'xnor', 'any' } ;
for k1 = 1:5
    op = ops {k1} ;
    fprintf ('\nop ( %s )=============================================\n', op) ;
    gtb_monoidinfo (ghb, [op '.logical']) ;
    gtb_monoidinfo (ghb, op, 'logical') ;
    nmonoids = nmonoids + 1 ;
end

% 6 complex
ops = { '+', '*', 'any' } ;
types = { 'single complex', 'double complex' } ;
for k1 = 1:3
    op = ops {k1} ;
    fprintf ('\nop ( %s )=============================================\n', op) ;
    for k2 = 1:2
        type = types {k2} ;
        gtb_monoidinfo (ghb, [op '.' type]) ;
        gtb_monoidinfo (ghb, op, type) ;
        nmonoids = nmonoids + 1 ;
    end
end

% 16 bitwise
ops = { 'bitor', 'bitand', 'bitxor', 'bitxnor' } ;
types = { 'uint8', 'uint16', 'uint32', 'uint64' } ;
for k1 = 1:4
    op = ops {k1} ;
    fprintf ('\nop ( %s )=============================================\n', op) ;
    for k2 = 1:4
        type = types {k2} ;
        gtb_monoidinfo (ghb, [op '.' type]) ;
        gtb_monoidinfo (ghb, op, type) ;
        nmonoids = nmonoids + 1 ;
    end
end

fprintf ('\n\n') ;
gtb_monoidinfo (ghb)

fprintf ('number of monoids: %d\n', nmonoids) ;
assert (nmonoids == 77) ;

fprintf ('gbtest53 (%d): all tests passed\n', ghb) ;

