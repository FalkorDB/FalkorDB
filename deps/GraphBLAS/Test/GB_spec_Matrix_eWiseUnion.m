function C = GB_spec_Matrix_eWiseUnion (C, Mask, accum, add, A, alpha, B, beta, descriptor, ignore)
%GB_SPEC_MATRIX_EWISEUNION a mimic of GxB_Matrix_eWiseUnion
%
% Usage:
% C = GB_spec_Matrix_eWiseUnion (C, Mask, accum, add, A, alpha, B, beta, descriptor)
%
% Computes C<Mask> = accum(C,T), in GraphBLAS notation, where T =A+B, A'+B,
% A+B' or A'+B'.  The pattern of T is the union of A and B.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || ~(nargin == 9 || nargin == 10))
    error ('usage: C = GB_spec_Matrix_eWiseUnion (C, Mask, accum, add, A, alpha, B, beta, descriptor)') ;
end

C = GB_spec_matrix (C) ;
A = GB_spec_matrix (A) ;
B = GB_spec_matrix (B) ;
[add_op optype ztype xtype ytype] = GB_spec_operator (add, C.class) ;
[C_replace Mask_comp Atrans Btrans Mask_struct] = ...
    GB_spec_descriptor (descriptor) ;
Mask = GB_spec_getmask (Mask, Mask_struct) ;

%-------------------------------------------------------------------------------
% do the work via a clean *.m interpretation of the entire GraphBLAS spec
%-------------------------------------------------------------------------------

alpha = full (alpha) ;
alpha = GB_mex_cast (alpha, xtype) ;

beta = full (beta) ;
beta = GB_mex_cast (beta, ytype) ;

% apply the descriptor to A
if (Atrans)
    A.matrix = A.matrix.' ;
    A.pattern = A.pattern' ;
end

% apply the descriptor to B
if (Btrans)
    B.matrix = B.matrix.' ;
    B.pattern = B.pattern' ;
end

% T = A+B, with typecasting
T.matrix = GB_spec_zeros (size (A.matrix), ztype) ;

if (GB_spec_is_positional (add))
    p = A.pattern | B.pattern ;
    [m, n] = size (A.matrix) ;
    for i = 1:m
        for j = 1:n
            if (p (i,j))
                T.matrix (i,j) = GB_spec_binop_positional (add_op, i, j, i, j) ;
            end
        end
    end
else
    % cast the entries into the class of the operator
    p = A.pattern & B.pattern ;
    A1 = GB_mex_cast (A.matrix (p), xtype) ;
    B1 = GB_mex_cast (B.matrix (p), ytype) ;
    % apply the operator
    T.matrix (p) = GB_spec_op (add, A1, B1) ;

    % for entries in A but not in B: T = add (A2, beta)
    p = A.pattern & ~B.pattern ;
    A2 = GB_mex_cast (A.matrix (p), xtype) ;
    beta = repmat (beta, length (A2), 1) ;
    % p
    % A2
    % beta
    T1 = GB_spec_op (add, A2, beta) ;
    T.matrix (p) = T1 ;

    % for entries in B but not in A: T = add (alpha, B2)
    p = B.pattern & ~A.pattern ;
    % p_full = full (p)
    B2 = GB_mex_cast (B.matrix (p), ytype) ;
    % B2
    alpha = repmat (alpha, length (B2), 1) ;
    % alpha
    T2 = GB_spec_op (add, alpha, B2) ;
    % T2
    T.matrix (p) = T2 ;
end

% the pattern of T is the union of both A and B
% A.pattern
% B.pattern
T.pattern = A.pattern | B.pattern ;

T.class = ztype ;

% C<Mask> = accum (C,T): apply the accum, then Mask, and return the result
C = GB_spec_accum_mask (C, Mask, accum, T, C_replace, Mask_comp, 0) ;


