function s = gb_isa (ghb, G, gtype, type)
%GB_ISA implements GrB/isa and GhB/isa.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~(ischar (type) || isstring (type)))
    error ('GrB:error', 'type must be a string') ;
end

if (isequal (type, 'GrB') || (ghb && isequal (type, 'GhB')))
    % GraphBLAS matrices have a class name of 'GrB' or 'GhB',
    % where GhB is a subclass of GrB.
    % If G is a GrB matrix, isa (G, 'GrB') is true but isa (G, 'GhB') is false.
    % If G is a GhB matrix, isa (G, 'GrB') and isa (G, 'GhB') are both true.
    % From the MATLAB documentation, isa (A, classname) is true if A is
    % instance of the classname OR a subclass of the classname.
    s = true ;
elseif isequal (type, 'numeric')
    % all GraphBLAS matrices are numeric
    s = true ;
elseif (isequal (type, 'float'))
    % GraphBLAS double, single, and complex matrices are 'float'
    s = gb_isfloat (gtype) ;
elseif (isequal (type, 'integer'))
    % GraphBLAS int* and uint* matrices are 'integer'
    s = gb_contains (gtype, 'int') ;
elseif (isequal (gtype, type))
    % specific cases, such as isa (G, 'double'), isa (G, 'int8'), etc
    s = true ;
else
    % catch-all for cases not handled above
    s = builtin ('isa', G, type) ;
end

