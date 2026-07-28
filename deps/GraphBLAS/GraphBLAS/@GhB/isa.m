function s = isa (G, type)
%ISA Determine if a GraphBLAS matrix is of specific type.
% For any GraphBLAS GhB matrix G, isa (G, 'GhB') and isa (G, 'numeric') are
% always true, even if G is logical, since many semirings are defined for that
% type.  Since GhB is a subclass of GrB, isa (G, 'GrB') is also true for a GhB
% matrix G.
%
% isa (G, 'float') is the same as isfloat (G), and is true if the matrix G has
% type 'double', 'single', 'single complex', or 'double complex'.
%
% isa (G, 'integer') is the same as isinteger (G), and is true if the matrix G
% has type 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', or
% 'uint64'.
%
% isa (G, type) is true if the type string matches the type of G.
%
% Otherwise, all other cases are handled with builtin ('isa',G,type).
%
% See also class, GhB.type, GhB/isnumeric, GhB/islogical, GhB/isfloat,
% GhB/isinteger, isobject, GhB/issparse, GhB/isreal.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = gb_isa (1, G, gb_type (G), type) ;

