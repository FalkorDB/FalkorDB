function c = grb_clear_coverage
%GRB_CLEAR_COVERAGE clear current statement coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

try
    global GraphBLAS_debug GraphBLAS_grbcov
    GraphBLAS_grbcov (:) = 0 ;
catch
end

