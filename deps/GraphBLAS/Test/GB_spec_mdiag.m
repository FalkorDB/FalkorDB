function C = GB_spec_mdiag (v, k, ctype)
%GB_SPEC_MDIAG a mimic of GxB_Matrix_diag
%
% Usage:
% C = GB_spec_mdiag (v, k, ctype)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

v = GB_spec_matrix (v) ;
if (nargin < 2)
    k = 0 ;
end
if (nargin < 3)
    ctype = v.class ;
end

%-------------------------------------------------------------------------------
% C = diag (v,k)
%-------------------------------------------------------------------------------

C.matrix  = diag (v.matrix, k) ;
C.pattern = diag (v.pattern, k) ;
C.class = ctype ;

