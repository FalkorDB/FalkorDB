function C = gb_power (A, B)
%GB_POWER .^ Array power.
% C = A.^B computes element-wise powers.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[am, an, atype] = gbsize (A) ;
[bm, bn, btype] = gbsize (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
a_is_real = ~gb_contains (atype, 'complex') ;
b_is_real = ~gb_contains (btype, 'complex') ;

% determine if C = A.^B is real or complex
if (a_is_real && b_is_real)
    % A and B are both real.  Determine if C might be complex.
    if (gb_contains (btype, 'int') || isequal (btype, 'logical'))
        % B is logical or integer, so C is real
        c_is_real = true ;
    elseif (gbisequal (B, gbapply ('round', B)))
        % B is floating point, but all values are equal to integers
        c_is_real = true ;
    elseif (gb_scalar (gbreduce ('min', A)) >= 0)
        % All entries in A are non-negative, so C is real
        c_is_real = true ;
    else
        % A has negative entries, and B is non-integer, so C can be complex.
        c_is_real = false ;
    end
else
    % A or B are complex, or both, so C must be complex
    c_is_real = false ;
end

if (c_is_real)
    % C is real
    ctype = gboptype (atype, btype) ;
else
    % C is complex
    if (gb_contains (atype, 'single') && gb_contains (btype, 'single'))
        ctype = 'single complex' ;
    else
        ctype = 'double complex' ;
    end
end

% B is always full
B = gbfull (B, ctype) ;

% determine the operator
op = ['pow.' ctype] ;

if (a_is_scalar)

    %----------------------------------------------------------------------
    % A is a scalar: C is a full matrix
    %----------------------------------------------------------------------

    C = gbapply2 (op, gbfull (A, ctype), B) ;

else

    %----------------------------------------------------------------------
    % A is a matrix
    %----------------------------------------------------------------------

    if (b_is_scalar)
        % A is a matrix, B is a scalar
        b = gb_scalar (B) ;
        if (b == 0)
            % special case:  C = A.^0 = ones (am, an, ctype)
            C = gb_scalar_to_full (am, an, ctype, gb_fmt (A), 1) ;
            return ;
        elseif (b == 1)
            % special case: C = A.^1 = A
            C = A ;
            return
        elseif (b <= 0)
            % 0.^b where b < 0 is Inf, so C is full
            C = gbapply2 (op, gbfull (A, ctype), B) ;
        else
            % The scalar b is > 0, and thus 0.^b is zero, so C is sparse.
            C = gbapply2 (op, A, B) ;
        end
    else
        % both A and B are matrices.  0.^0 is 1, so C is full.
        C = gbemult (op, gbfull (A, ctype), B) ;
    end

end

% convert C to real if imaginary part is zero
if (~c_is_real)
    C = gb_check_imag_zero (C) ;
end

