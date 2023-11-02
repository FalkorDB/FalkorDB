function c = grb_get_coverage
%GRB_GET_COVERAGE return current statement coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

c = 0 ;
try
    global GraphBLAS_debug GraphBLAS_grbcov
    c = sum (GraphBLAS_grbcov > 0) ;
catch
end

