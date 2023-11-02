function C = single (G)
%SINGLE cast a GraphBLAS matrix to built-in full single matrix.
% C = single (G) typecasts the GrB matrix G to a built-in full single
% matrix.  The result C is full since sparse single matrices are not
% built-in.  C is real if G is real, and complex if G is complex.
%
% To typecast the matrix G to a GraphBLAS sparse single matrix instead,
% use C = GrB (G, 'single').  To typecast to a sparse single complex
% matrix, use G = GrB (G, 'single complex').
%
% See also GrB, GrB/double, GrB/complex, GrB/logical, GrB/int8, GrB/int16,
% GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32, GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
desc.kind = 'full' ;
if (gb_contains (gbtype (G), 'complex'))
    z = complex (single (0)) ;
    ctype = 'single complex' ;
else
    z = single (0) ;
    ctype = 'single' ;
end

C = gbfull (G, ctype, z, desc) ;                % export as a full matrix

