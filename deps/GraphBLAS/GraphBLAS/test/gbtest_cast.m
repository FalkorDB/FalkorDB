function C = gbtest_cast (A, type)
%GBTEST_CAST cast a built-in matrix to another type.
% C = gbtest_cast (A, type) is identical to C = cast (A, type) when type
% is a valid built-in class ('logical', 'int8', 'int16', 'int32', 'int64',
% 'uint8', 'uint16', 'uint32', 'uint64', 'single', 'double').  Otherwise,
% A is converted to a single complex or double complex matrix C.
%
% A must be a full built-in matrix (not sparse).  C is a full built-in matrix.
% To cast any matrix to a GraphBLAS matrix instead, use C = GrB (A, type).
%
% See also cast.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (issparse (A))
    error ('A must be full') ;
end
if (isa (A, 'GrB'))
    error ('A must be a built-in matrix') ;
end

if (gb_contains (type, 'complex'))
    if (gb_contains (type, 'single'))
        C = complex (single (A)) ;
    else
        C = complex (double (A)) ;
    end
else
    C = cast (A, type) ;
end

assert (~issparse (C)) ;
assert (~isa (C, 'GrB')) ;

