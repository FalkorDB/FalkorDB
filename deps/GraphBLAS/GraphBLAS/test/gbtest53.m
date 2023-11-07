function gbtest53
%GBTEST53 test GrB.monoidinfo

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

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
        GrB.monoidinfo ([op '.' type]) ;
        GrB.monoidinfo (op, type) ;
        nmonoids = nmonoids + 1 ;
    end
end

% 5 boolean monoids:
ops = { '|', '&', 'xor', 'xnor', 'any' } ;
for k1 = 1:5
    op = ops {k1} ;
    fprintf ('\nop ( %s )=============================================\n', op) ;
    GrB.monoidinfo ([op '.logical']) ;
    GrB.monoidinfo (op, 'logical') ;
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
        GrB.monoidinfo ([op '.' type]) ;
        GrB.monoidinfo (op, type) ;
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
        GrB.monoidinfo ([op '.' type]) ;
        GrB.monoidinfo (op, type) ;
        nmonoids = nmonoids + 1 ;
    end
end

fprintf ('\n\n') ;
GrB.monoidinfo

fprintf ('number of monoids: %d\n', nmonoids) ;
assert (nmonoids == 77) ;

fprintf ('gbtest53: all tests passed\n') ;

