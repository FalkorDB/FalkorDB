function s = isa (G, type)
%ISA Determine if a GraphBLAS matrix is of specific type.
% For any GraphBLAS matrix G, isa (G, 'GrB') and isa (G, 'numeric') are
% always true, even if G is logical, since many semirings are defined for
% that type.
%
% isa (G, 'float') is the same as isfloat (G), and is true if the GrB
% matrix G has type 'double', 'single', 'single complex', or 'double
% complex'.
%
% isa (G, 'integer') is the same as isinteger (G), and is true if the GrB
% matrix G has type 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
% 'uint32', or 'uint64'.
%
% isa (G, type) is true if the type string matches the type of G.
%
% Otherwise, all other cases are handled with builtin ('isa',G,type).
%
% See also class, GrB.type, GrB/isnumeric, GrB/islogical, GrB/isfloat,
% GrB/isinteger, isobject, GrB/issparse, GrB/isreal.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isequal (type, 'GrB') || isequal (type, 'numeric'))
    % all GraphBLAS matrices are numeric, and have class name 'GrB'
    s = true ;
elseif (isequal (type, 'float'))
    % GraphBLAS double, single, and complex matrices are 'float'
    s = isfloat (G) ;
elseif (isequal (type, 'integer'))
    % GraphBLAS int* and uint* matrices are 'integer'
    s = isinteger (G) ;
elseif (isequal (GrB.type (G), type))
    % specific cases, such as isa (G, 'double'), isa (G, 'int8'), etc
    s = true ;
else
    % catch-all for cases not handled above
    s = builtin ('isa', G, type) ;
end

