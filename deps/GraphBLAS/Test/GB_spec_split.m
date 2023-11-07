function C = GB_spec_split (A, ms, ns)
%GB_SPEC_SPLIT a mimic of GxB_Matrix_split
%
% Usage:
% C = GB_spec_split (A, ms, ns)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

A = GB_spec_matrix (A) ;
atype = A.class ;

%-------------------------------------------------------------------------------
% C = split (A, ms, ns)
%-------------------------------------------------------------------------------

C_matrix  = mat2cell (A.matrix,  ms, ns) ;
C_pattern = mat2cell (A.pattern, ms, ns) ;
C = cell (length (ms), length (ns)) ;

for i = 1:length(ms)
    for j = 1:length(ns)
        clear T
        T.matrix  = C_matrix  {i,j} ;
        T.pattern = C_pattern {i,j} ;
        T.class   = atype ;
        C {i,j} = T ;
    end
end

