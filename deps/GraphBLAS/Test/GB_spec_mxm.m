function C = GB_spec_mxm (C, Mask, accum, semiring, A, B, descriptor)
%GB_SPEC_MXM a mimic of GrB_mxm
%
% Usage:
% C = GB_spec_mxm (C, Mask, accum, semiring, A, B, descriptor)
%
% Computes C<Mask> = accum(C,T), in GraphBLAS notation, where T =A*B, A'*B,
% A*B' or A'*B'.  The matrix C is returned as a struct with C.matrix being the
% values of the matrix, C.pattern the 'nonzero' pattern, and C.class the type
% of the matrix.  A and B can be plain matrices on input, or they can be
% structs like C.  See GB_spec_matrix.m for more details.  See
% GB_spec_transpose.m for a description of the input parameters, except for
% semiring; see GB_spec_semiring for a description of the semiring.
%
% The semiring defines the multiply and add operators and the additive
% identity.  The matrix multiplication T = A*B is then defined as:
%
%   [m s] = size (A) ;
%   [s n] = size (B) ;
%   ztype, xtype, ytype: the types of the multiply operator
%   T = identity (m,n,ztype) ; where identity is defined by semiring.add
%   for j = 1:n
%       for i = 1:m
%           for k = 1:s
%               if (A (i,k) and B (k,j) are 'nonzero'
%                   aik = cast (A(i,k), xtype)
%                   bkj = cast (B(k,j), ytype)
%                   T (i,j) = add (T (i,j), multiply (aik, bkj))
%               end
%           end
%       end
%   end
%
% Where A(i,j) 'nonzero' means that the entry is in the data structure for the
% sparse matrix A.  If it is not there, it is implied to be equal to the
% addititive identity.  The identity value does not appear in the sparse
% GraphBLAS matrix; it is the value of entries not present in the data
% structure.  The GB_spec_*.m functions operate on dense matrices,
% however, so these entries must be explicitly set to the additive identity.
%
% This gives a matrix T which is then accumulated into the result via
% C<Mask> = accum (C,T).  See GrB_accum_mask for a description of this
% last step.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || nargin ~= 7)
    error ('usage: C = GB_spec_mxm (C, Mask, accum, semiring, A, B, descriptor)') ;
end

% Convert inputs to dense matrices with explicit patterns and types,
% and with where X(~X.pattern)==identity for all matrices A, B, and C.
[multiply add identity ztype xtype ytype] = GB_spec_semiring (semiring) ;
if (isempty (identity))
    identity = 0 ;
end
C = GB_spec_matrix (C, identity) ;
A = GB_spec_matrix (A, identity) ;
B = GB_spec_matrix (B, identity) ;
[C_replace Mask_comp Atrans Btrans Mask_struct] = ...
    GB_spec_descriptor (descriptor) ;
Mask = GB_spec_getmask (Mask, Mask_struct) ;

%-------------------------------------------------------------------------------
% do the work via a clean *.m interpretation of the entire GraphBLAS spec
%-------------------------------------------------------------------------------

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

% T = A*B
[m s] = size (A.matrix) ;
[s n] = size (B.matrix) ;
T.matrix = GB_spec_zeros ([m n], ztype) ;
T.pattern = zeros (m, n, 'logical') ;
T.matrix (:,:) = identity ;
T.class = ztype ;

A_matrix = GB_mex_cast (A.matrix, xtype) ;
B_matrix = GB_mex_cast (B.matrix, ytype) ;

op_is_positional = GB_spec_is_positional (multiply) ;
multop = multiply.opname ;

for j = 1:n
    for i = 1:m
        for k = 1:s
            % T (i,j) += A (i,k) * B (k,j), using the semiring
            if (A.pattern (i,k) && B.pattern (k,j))
                if (op_is_positional)
                    z = GB_spec_binop_positional (multop, i, k, k, j) ;
                else
                    z = GB_spec_op (multiply, A_matrix (i,k), B_matrix (k,j)) ;
                end
                T.matrix (i,j) = GB_spec_op (add, T.matrix (i,j), z) ;
                T.pattern (i,j) = true ;
            end
        end
    end
end

% C<Mask> = accum (C,T): apply the accum, then Mask, and return the result
C = GB_spec_accum_mask (C, Mask, accum, T, C_replace, Mask_comp, identity) ;

