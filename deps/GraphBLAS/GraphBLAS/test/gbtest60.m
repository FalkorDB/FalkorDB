function gbtest60 (ghb)
%GBTEST60 test [GrB,GhB].issigned

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

% 8 signed types:
signed_types   = { 'double', 'single', ...
    'int8', 'int16', 'int32', 'int64', ...
    'single complex', 'double complex' } ;

% 5 unsigned types:
unsigned_types = { 'logical', 'uint8', 'uint16', 'uint32', 'uint64' } ;

for k = 1:length (signed_types)
    type = signed_types {k} ;
    assert (gtb_issigned (ghb, type)) ;
    G = gtb (ghb, 1, type) ;
    assert (gtb_issigned (ghb, G)) ;
    if (isequal (type, 'single complex'))
        A = complex (single (pi)) ;
    elseif (isequal (type, 'double complex'))
        A = complex (double (pi)) ;
    else
        A = cast (pi, type) ;
    end
    assert (gtb_issigned (ghb, A)) ;
end

for k = 1:length (unsigned_types)
    type = unsigned_types {k} ;
    assert (~gtb_issigned (ghb, type)) ;
    G = gtb (ghb, 1, type) ;
    assert (~gtb_issigned (ghb, G)) ;
    A = cast (1, type) ;
    assert (~gtb_issigned (ghb, A)) ;
end

fprintf ('gbtest60 (%d): all tests passed\n', ghb) ;

