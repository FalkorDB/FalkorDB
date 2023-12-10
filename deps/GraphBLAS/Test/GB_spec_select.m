function C = GB_spec_select (C, Mask, accum, opname, A, thunk, descriptor)
%GB_SPEC_SELECT a mimic of GxB_select
%
% Usage:
% C = GB_spec_select (C, Mask, accum, opname, A, thunk, descriptor)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || nargin ~= 7)
    error ('usage: C = GB_spec_select (C, Mask, accum, opname, A, thunk, desc)');
end

C = GB_spec_matrix (C) ;
A = GB_spec_matrix (A) ;
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
thunk = full (thunk) ;
athunk = GB_mex_cast (thunk, atype) ;

is_complex = test_contains (atype, 'complex') ;
if (is_complex)
    switch (opname)
        case { 'gt_zero', 'ge_zero', 'lt_zero', 'le_zero', ...
               'gt_thunk', 'ge_thunk', 'lt_thunk', 'le_thunk' }
            error ('op %s not defined for complex types', opname) ;
        otherwise
            % op is OK
    end
end

switch (opname)
    case 'tril'
        p = tril (A.pattern, thunk) ;
    case 'triu'
        p = triu (A.pattern, thunk) ;
    case 'diag'
        p = tril (triu (A.pattern, thunk), thunk) ;
    case 'offdiag'
        p = tril (A.pattern, thunk-1) | triu (A.pattern, thunk+1) ;
    case 'nonzero'
        p = A.pattern & (A.matrix ~= 0) ;
    case 'eq_zero'
        p = A.pattern & (A.matrix == 0) ;
    case 'gt_zero'
        p = A.pattern & (A.matrix > 0) ;
    case 'ge_zero'
        p = A.pattern & (A.matrix >= 0) ;
    case 'lt_zero'
        p = A.pattern & (A.matrix < 0) ;
    case 'le_zero'
        p = A.pattern & (A.matrix <= 0) ;
    case 'ne_thunk'
        p = A.pattern & (A.matrix ~= athunk) ;
    case 'eq_thunk'
        p = A.pattern & (A.matrix == athunk) ;
    case 'gt_thunk'
        p = A.pattern & (A.matrix > athunk) ;
    case 'ge_thunk'
        p = A.pattern & (A.matrix >= athunk) ;
    case 'lt_thunk'
        p = A.pattern & (A.matrix < athunk) ;
    case 'le_thunk'
        p = A.pattern & (A.matrix <= athunk) ;
    case 'isnan'
        p = A.pattern & isnan (A.matrix) ;
    otherwise
        error ('invalid op') ;
end

T.matrix (p) = A.matrix (p) ;
T.pattern = p ;

% C<Mask> = accum (C,T): select the accum, then Mask, and return the result
C = GB_spec_accum_mask (C, Mask, accum, T, C_replace, Mask_comp, 0) ;

