function [s, ztype] = GB_spec_is_idxunop (op, optype)
%GB_SPEC_IS_IDXUNOP determine if an op is an idxunop

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isstruct (op))
    op = op.opname ;
end

if (nargin < 2)
    optype = 'int64' ;
end

switch (op)
    case { 'rowindex', 'colindex', 'diagindex' }
        s = strcmp (optype, 'int64') || strcmp (optype, 'int32') ;
        ztype = optype ;
    case { 'tril', 'triu', 'diag', 'offdiag', ...
        'colle', 'colgt', 'rowle', 'rowgt' }
        s = strcmp (optype, 'int64') ;
        ztype = 'logical' ;
    case { 'valuene', 'valueeq' }
        s = true ;
        ztype = 'logical' ;
    case { 'valuelt', 'valuele', 'valuegt', 'valuege' }
        s = ~test_contains (optype, 'complex') ;
        ztype = 'logical' ;
    otherwise
        s = false ;
        ztype = '' ;
end

if (~s)
    ztype = '' ;
end

