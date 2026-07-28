function C = reduce (arg1, arg2, arg3, arg4, arg5)
%GHB.REDUCE reduce a matrix to a scalar.
%
% syntax for a new matrix C:                    computation:
% c = GhB.reduce (op, A, desc)                  % c = op(A)
% c = GhB.reduce (cin, op, A, desc)             % c = cin ; c = op(A)
% c = GhB.reduce (cin, accum, op, A, desc)      % c = cin ; c += op(A)
%
% in-place syntax:
% GhB.reduce (c, op, A, desc)                   % c = op(A)
% GhB.reduce (c, accum, op, A, desc)            % c += op(A)
%
% GhB.reduce reduces a matrix to a scalar, using the given op as a monoid:
%
%   Monoids for real non-logical types: '+', '*', 'max', 'min', 'any'
%   For logical: '|', '&', 'xor', 'eq', 'any'
%   For complex types: '+', '*', 'any'
%   For integer types: 'bitor', 'bitand', 'bitxor', 'bitxnor'
%
% See 'help GrB.monoidinfo' for more details on the available monoids.
%
% The op and A arguments are required.  All others are optional.  The op is
% applied to all entries of the matrix A to reduce them to a single scalar
% result.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
% For the in-place syntax, the GhB scalar c is modified in-place.
%
% cin: an optional input scalar into which the result can be accumulated
% with c = accum (cin, result).
%
% See also GhB.vreduce, GhB/sum, GhB/prod, GhB/max, GhB/min, GrB.monoidinfo,
% GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargout == 0)
    narginchk (3, 5) ;
else
    narginchk (2, 5) ;
end

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (nargin >= 3 && gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (nargin >= 4 && gb_is_grb (arg4))
    arg4 = struct (arg4) ;
end

% arg5: if present, it must be the descriptor

if (nargout == 0)
    switch (nargin)
        case 3
            gbmex_reduce (1, arg1, arg2, arg3) ;
        case 4
            gbmex_reduce (1, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_reduce (1, arg1, arg2, arg3, arg4, arg5) ;
    end
else
    switch (nargin)
        case 2
            [C_opaque, kind] = gbmex_reduce (1, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_reduce (1, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_reduce (1, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_reduce (1, arg1, arg2, arg3, arg4, arg5) ;
    end
    C = gb_mexfunction_result (1, C_opaque, kind) ;
end

