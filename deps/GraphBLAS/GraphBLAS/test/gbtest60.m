function gbtest60
%GBTEST60 test GrB.issigned

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% 8 signed types:
signed_types   = { 'double', 'single', ...
    'int8', 'int16', 'int32', 'int64', ...
    'single complex', 'double complex' } ;

% 5 unsigned types:
unsigned_types = { 'logical', 'uint8', 'uint16', 'uint32', 'uint64' } ;

for k = 1:length (signed_types)
    type = signed_types {k} ;
    assert (GrB.issigned (type)) ;
    G = GrB (1, type) ;
    assert (GrB.issigned (G)) ;
    if (isequal (type, 'single complex'))
        A = complex (single (pi)) ;
    elseif (isequal (type, 'double complex'))
        A = complex (double (pi)) ;
    else
        A = cast (pi, type) ;
    end
    assert (GrB.issigned (A)) ;
end

for k = 1:length (unsigned_types)
    type = unsigned_types {k} ;
    assert (~GrB.issigned (type)) ;
    G = GrB (1, type) ;
    assert (~GrB.issigned (G)) ;
    A = cast (1, type) ;
    assert (~GrB.issigned (A)) ;
end

fprintf ('gbtest60: all tests passed\n') ;

