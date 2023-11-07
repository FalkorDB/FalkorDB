function V = GB_spec_vdiag (A, k, vtype)
%GB_SPEC_VDIAG a mimic of GxB_Vector_diag
%
% Usage:
% V = GB_spec_vdiag (A, k, vtype)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

A = GB_spec_matrix (A) ;
if (nargin < 2)
    k = 0 ;
end
if (nargin < 3)
    vtype = A.class ;
end

%-------------------------------------------------------------------------------
% v = diag (A,k)
%-------------------------------------------------------------------------------

V.matrix  = diag (A.matrix, k) ;
V.pattern = diag (A.pattern, k) ;
V.class = vtype ;

