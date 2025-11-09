function ok = GB_spok (A)
%GB_SPOK check if a matrix is valid

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (issparse (A))
    ok = spok (A) ;
else
    ok = true ;
end
