function z = GB_spec_idxunop (opname, aij, i, j, thunk)
%GB_SPEC_IDXUNOP apply an idxunop

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% convert indices to 0-based
i = i - 1 ;
j = j - 1 ;

switch (opname)
    case { 'rowindex' }
        z = i + thunk ;
    case { 'colindex' }
        z = j + thunk ;
    case { 'diagindex' }
        z = j - (i + thunk) ;
    case { 'tril' }
        z = (j <= (i+thunk)) ;
    case { 'triu' }
        z = (j >= (i+thunk)) ;
    case { 'diag' }
        z = (j == (i+thunk)) ;
    case { 'offdiag' }
        z = (j ~= (i+thunk)) ;
    case { 'colle' }
        z = (j <= thunk) ;
    case { 'colgt' }
        z = (j > thunk) ;
    case { 'rowle' }
        z = (i <= thunk) ;
    case { 'rowgt' }
        z = (i > thunk) ;
    case { 'valuene' }
        z = (aij ~= thunk) ;
    case { 'valueeq' }
        z = (aij == thunk) ;
    case { 'valuelt' }
        z = (aij < thunk) ;
    case { 'valuele' }
        z = (aij <= thunk) ;
    case { 'valuegt' }
        z = (aij > thunk) ;
    case { 'valuege' }
        z = (aij >= thunk) ;
end

