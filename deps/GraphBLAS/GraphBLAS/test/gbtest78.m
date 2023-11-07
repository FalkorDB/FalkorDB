function gbtest78
%GBTEST78 test integer operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

A = uint8 (magic (4)) ;
A = A (:,1:3) ;
A (1,1) = 0 ;

assert (GrB.isbycol (A)) ;
assert (~GrB.isbyrow (A)) ;

disp (A, GrB (5)) ;

G = GrB (A) ;

C = (G < -1) ;
assert (isequal (C, sparse (false (4,3)))) ;

C = (-1 < G) ;
assert (isequal (C, sparse (true (4,3)))) ;

C = GrB.empty ;
assert (isequal (C, [ ])) ;

C1 = bitset (A, 1, 1) ;
C2 = bitset (G, 1, GrB (1)) ;
assert (isequal (C1, C2)) ;

C1 = bitshift (uint64 (3), A) ;
C2 = bitshift (uint64 (3), G) ;
assert (isequal (C1, C2)) ;

types = {
    'double'
    'int8'
    'int16'
    'int32'
    'int64'
    'uint8'
    'uint16'
    'uint32'
    'uint64'
    } ;

% bitset (a,b,V) where a and b are scalars
V = magic (4) .* mod (magic (4), 2) ;
G = GrB (V) ;
for k = 1:length (types)
    type = types {k} ;
    fprintf ('%s ', type) ;
    for a = 0:8
        for b = 1:8
            A = cast (a, type) ;
            B = cast (b, type) ;
            C1 = bitset (A, B, V) ;
            C2 = bitset (A, B, G) ;
            assert (isequal (C1, C2)) ;
        end
    end
end

fprintf ('\ngbtest78: all tests passed\n') ;

