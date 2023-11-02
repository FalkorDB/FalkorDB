function C = GB_spec_select_idxunop (C, Mask, accum, op, A, thunk, descriptor)
%GB_SPEC_SELECT_IDXUNOP a mimic of GrB_select
%
% Usage:
% C = GB_spec_select_idxunop (C, Mask, accum, op, A, thunk, descriptor)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || nargin ~= 7)
    error ('usage: C = GB_spec_select_idxunop (C, Mask, accum, op, A, thunk, desc)');
end

C = GB_spec_matrix (C) ;
A = GB_spec_matrix (A) ;
[opname optype ztype xtype ytype] = GB_spec_operator (op, C.class) ;
[C_replace Mask_comp Atrans Btrans Mask_struct] = ...
    GB_spec_descriptor (descriptor) ;
Mask = GB_spec_getmask (Mask, Mask_struct) ;

%-------------------------------------------------------------------------------
% do the work via a clean *.m interpretation of the entire GraphBLAS spec
%-------------------------------------------------------------------------------

% select the descriptor to A
if (Atrans)
    A.matrix = A.matrix.' ;
    A.pattern = A.pattern' ;
end

atype = A.class ;
T.matrix = GB_spec_zeros (size (A.matrix), atype) ;

X = A.matrix ;
if (~isequal (atype, xtype) && ~isempty (xtype))
    X = GB_mex_cast (X, xtype) ;
end

if (isstruct (thunk))
    thunk = GB_mex_cast (thunk.matrix, thunk.class) ;
end
thunk = full (thunk) ;
thunk = GB_mex_cast (thunk, ytype) ;
ithunk = GB_mex_cast (thunk, 'int64') ;

[m n] = size (A.matrix) ;
I = int64 (repmat ((0:m-1)', 1, n)) ;
J = int64 (repmat ((0:n-1),  m, 1)) ;

switch (opname)
    case { 'rowindex' }
        p = A.pattern & ((I + ithunk) ~= 0) ;
    case { 'colindex' }
        p = A.pattern & ((J + ithunk) ~= 0) ;
    case { 'diagindex' }
        p = A.pattern & ((J - (I + ithunk)) ~= 0) ;
    case { 'tril' }
        p = A.pattern & (J <= (I + ithunk)) ;
    case { 'triu' }
        p = A.pattern & (J >= (I + ithunk)) ;
    case { 'diag' }
        p = A.pattern & (J == (I + ithunk)) ;
    case { 'offdiag' }
        p = A.pattern & (J ~= (I + ithunk)) ;
    case { 'colle' }
        p = A.pattern & (J <= ithunk) ;
    case { 'colgt' }
        p = A.pattern & (J > ithunk) ;
    case { 'rowle' }
        p = A.pattern & (I <= ithunk) ;
    case { 'rowgt' }
        p = A.pattern & (I > ithunk) ;
    case { 'valuene' }
        p = A.pattern & (X ~= thunk) ;
    case { 'valueeq' }
        p = A.pattern & (X == thunk) ;
    case { 'valuelt' }
        p = A.pattern & (X < thunk) ;
    case { 'valuele' }
        p = A.pattern & (X <= thunk) ;
    case { 'valuegt' }
        p = A.pattern & (X > thunk) ;
    case { 'valuege' }
        p = A.pattern & (X >= thunk) ;
    otherwise
        error ('invalid op') ;
end
    
T.matrix (p) = A.matrix (p) ;
T.pattern = p ;

% C<Mask> = accum (C,T): select the accum, then Mask, and return the result
C = GB_spec_accum_mask (C, Mask, accum, T, C_replace, Mask_comp, 0) ;

