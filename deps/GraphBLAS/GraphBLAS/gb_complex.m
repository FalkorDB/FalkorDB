function C = gb_complex (A, B)
%GB_COMPLEX: implements GrB.complex and GhB.complex.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin == 1)

    % with a single input, A must be a GraphBLAS matrix (otherwise,
    % this overloaded method for GrB objects would not be called).
    % Convert A to a built-in double complex matrix C.
    C = gzb_cast (A, 'double complex') ;

else

    % with two inputs, A and B are real matrices (GrB, GhB, or built-in)
    % but at least one must be GrB or otherwise this overloaded method
    % would not be called).  The output is a double complex matrix.

    if (gb_is_grb (B))
        B = struct (B) ;
    end

    [am, an, atype] = gbmex_size (A) ;
    [bm, bn, btype] = gbmex_size (B) ;
    a_is_scalar = (am == 1) && (an == 1) ;
    b_is_scalar = (bm == 1) && (bn == 1) ;

    if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
        error ('GrB:error', 'inputs must be real') ;
    end

    if (a_is_scalar)
        if (b_is_scalar)
            % both A and B are scalars.  C is also a scalar.
            a = gzb_full (1, A, 'double') ;
            b = gzb_full (1, B, 'double') ;
            desc.kind = 'full' ;
            C = gzb_emult (1, 'cmplx.double', a, b, desc) ;
        else
            % A is a scalar, B is a matrix.  C is full, unless A == 0.
            if (gb_scalar (A) == 0)
                % C = 1i*B, so A = zero, C is sparse or full.
                desc.kind = 'builtin' ;
                C = gzb_apply2 (1, 'cmplx.double', 0, B, desc) ;
            else
                % expand A and B to full double matrices; C is full
                desc.kind = 'full' ;
                a = gb_scalar_to_full (1, bm, bn, 'double', gb_fmt (B), A) ;
                b = gzb_full (1, B, 'double') ;
                C = gzb_emult (1, 'cmplx.double', a, b, desc) ;
            end
        end
    else
        if (b_is_scalar)
            % A is a matrix, B is a scalar.  C is full, unless B == 0.
            if (gb_scalar (B) == 0)
                % C = complex (A); C is sparse or full
                C = gzb_cast (A, 'double.complex') ;
            else
                % expand A and B to full double matrices; C is full
                desc.kind = 'full' ;
                a = gzb_full (1, A, 'double') ;
                b = gb_scalar_to_full (1, am, an, 'double', gb_fmt (A), B) ;
                C = gzb_emult (1, 'cmplx.double', a, b, desc) ;
            end
        else
            % both A and B are matrices.  C is sparse or full.
            desc.kind = 'builtin' ;
            b = gzb_apply2 (1, B, '*', 1i) ;
            C = gzb_eadd (1, A, '+', b, desc) ;
        end
    end

end

% return C as a builtin MATLAB/Octave matrix
C = gb_builtin (C) ;

