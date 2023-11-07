function C = complex (A, B)
%COMPLEX cast to a built-in double complex matrix.
% C = complex (G) typecasts the GraphBLAS matrix G to into a built-in
% double complex matrix.  C is full if all entries in G are present,
% or sparse otherwse.
%
% With two inputs, C = complex (A,B) returns a matrix C = A + 1i*B,
% where A or B are real matrices (@GrB/built-in in any
% combination).  If A or B are nonzero scalars and the other input is a
% matrix, or if both A and B are scalars, C is full.
%
% To typecast the matrix G to a GraphBLAS double complex matrix
% instead, use C = GrB (G, 'complex') or C = GrB (G, 'double complex').
% To typecast the matrix G to a GraphBLAS single complex matrix, use
% C = GrB (G, 'single complex').
%
% To construct a complex GraphBLAS matrix from real GraphBLAS matrices
% A and B, use C = A + 1i*B instead.
%
% Since sparse single complex matrices are not built-in, C is
% always returned as a double complex matrix (sparse or full).
%
% See also cast, GrB, GrB/double, GrB/single, GrB/logical, GrB/int8,
% GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)

    % with a single input, A must be a GraphBLAS matrix (otherwise,
    % this overloaded method for GrB objects would not be called).
    % Convert A to a built-in double complex matrix C.
    A = A.opaque ;
    C = gbbuiltin (A, 'double complex') ;

else

    % with two inputs, A and B are real matrices (@GrB or built-in)
    % but at least one must be GrB or otherwise this overloaded method
    % would not be called).  The output is a double complex matrix.
    if (isobject (A))
        A = A.opaque ;
    end

    if (isobject (B))
        B = B.opaque ;
    end

    [am, an, atype] = gbsize (A) ;
    [bm, bn, btype] = gbsize (B) ;
    a_is_scalar = (am == 1) && (an == 1) ;
    b_is_scalar = (bm == 1) && (bn == 1) ;

    if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
        error ('GrB:error', 'inputs must be real') ;
    end

    if (a_is_scalar)
        if (b_is_scalar)
            % both A and B are scalars.  C is also a scalar.
            A = gbfull (A, 'double') ;
            B = gbfull (B, 'double') ;
            desc.kind = 'full' ;
            C = gbemult ('cmplx.double', A, B, desc) ;
        else
            % A is a scalar, B is a matrix.  C is full, unless A == 0.
            if (gb_scalar (A) == 0)
                % C = 1i*B, so A = zero, C is sparse or full.
                desc.kind = 'builtin' ;
                C = gbapply2 ('cmplx.double', 0, B, desc) ;
            else
                % expand A and B to full double matrices; C is full
                A = gb_scalar_to_full (bm, bn, 'double', gb_fmt (B), A) ;
                B = gbfull (B, 'double') ;
                desc.kind = 'full' ;
                C = gbemult ('cmplx.double', A, B, desc) ;
            end
        end
    else
        if (b_is_scalar)
            % A is a matrix, B is a scalar.  C is full, unless B == 0.
            if (gb_scalar (B) == 0)
                % C = complex (A); C is sparse or full
                C = gbbuiltin (A, 'double.complex') ;
            else
                % expand A and B to full double matrices; C is full
                A = gbfull (A, 'double') ;
                B = gb_scalar_to_full (am, an, 'double', gb_fmt (A), B) ;
                desc.kind = 'full' ;
                C = gbemult ('cmplx.double', A, B, desc) ;
            end
        else
            % both A and B are matrices.  C is sparse or full.
            desc.kind = 'builtin' ;
            C = gbeadd (A, '+', gbapply2 (B, '*', 1i), desc) ;
        end
    end

end

