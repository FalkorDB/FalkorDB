function [I, J, X] = GB_spec_extractTuples (A, xclass)
%GB_SPEC_EXTRACTTUPLES a mimic of GrB_*_extractTuples

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

A = GB_spec_matrix (A) ;
if (nargin < 2)
    xclass = A.class ;
end
p = A.pattern ;
[I J] = find (p) ;
I = uint64 (I-1) ;
J = uint64 (J-1) ;
X = GB_mex_cast (A.matrix (p), xclass) ;

I = I (:) ;
J = J (:) ;
X = X (:) ;


