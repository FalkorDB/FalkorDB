function C = vreduce (arg1, arg2, arg3, arg4, arg5, arg6)
%GHB.VREDUCE reduce a matrix to a vector.
%
% syntax for a new vector C:                    computation:
% C = GhB.vreduce (op, A, desc)                 % C = op(A)
% C = GhB.vreduce (Cin, op, A, desc)            % C = Cin ; C = op(A)
% C = GhB.vreduce (Cin, accum, op, A, desc)     % C = Cin ; C += op(A)
%
% in-place syntax:
% GhB.vreduce (C, op, A, desc)                  % C = op(A)
% GhB.vreduce (C, accum, op, A, desc)           % C += op(A)
%
% GhB.reduce reduces a matrix to a scalar, using the given op as a monoid,
% where T=op(A) computes T(i) = sum(A(i,:)) by default, where "sum" is the
% application of the given op.
%
%   Monoids for real non-logical types: '+', '*', 'max', 'min', 'any'
%   For logical: '|', '&', 'xor', 'eq', 'any'
%   For complex types: '+', '*', 'any'
%   For integer types: 'bitor', 'bitand', 'bitxor', 'bitxnor'
%
% See 'help GrB.monoidinfo' for more details on the available monoids.
%
% By default, each row of A is reduced to a scalar.  If Cin is not present, C
% (i) = reduce (A (i,:)).  In this case, Cin and C are column vectors of size
% m-by-1, where A is m-by-n.  If desc.in0 is 'transpose', then A.' is reduced
% to a column vector; C (j) = reduce (A (:,j)).  In this case, Cin and C are
% column vectors of size n-by-1, if A is m-by-n.  See 'help GrB.descriptorinfo'
% for more details.
%
% The op and A arguments are required.  All others are optional.  The op is
% applied to all entries in each or or column of the matrix A to reduce them to
% a single scalar result.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
% For the in-place syntax, the GhB scalar c is modified in-place.
%
% Cin or C: an optional input vector into which the result can be accumulated.
%
% See also GhB.vreduce, GhB/sum, GhB/prod, GhB/max, GhB/min, GrB.monoidinfo,
% GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargout == 0)
    narginchk (3, 6) ;
else
    narginchk (2, 6) ;
end

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (nargin >= 4 && gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (nargin >= 4 && gb_is_grb (arg4))
    arg4 = struct (arg4) ;
end

if (nargin >= 5 && gb_is_grb (arg5))
    arg5 = struct (arg5) ;
end

% arg6: if present, it must be the descriptor

if (nargout == 0)
    switch (nargin)
        case 3
            gbmex_vreduce (1, arg1, arg2, arg3) ;
        case 4
            gbmex_vreduce (1, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_vreduce (1, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            gbmex_vreduce (1, arg1, arg2, arg3, arg4, arg5, arg6) ;
    end
else
    switch (nargin)
        case 2
            [C_opaque, kind] = gbmex_vreduce (1, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_vreduce (1, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_vreduce (1, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_vreduce (1, arg1, arg2, arg3, arg4, ...
                arg5) ;
        case 6
            [C_opaque, kind] = gbmex_vreduce (1, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
    end
    C = gb_mexfunction_result (1, C_opaque, kind) ;
end

