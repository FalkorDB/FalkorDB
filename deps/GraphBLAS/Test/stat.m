function [d nthreads] = stat
%STAT report status of statement coverage and malloc debugging

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

global GraphBLAS_debug GraphBLAS_grbcov

if (isempty (GraphBLAS_debug))
    GraphBLAS_debug = false ;
end

nthreads = nthreads_get ;

if (nargout == 0)
    fprintf ('malloc debug: %d  nthreads %d\n', GraphBLAS_debug, nthreads) ;
else
    d = GraphBLAS_debug ;
end

if (~isempty (GraphBLAS_grbcov))
    covered = sum (GraphBLAS_grbcov > 0) ;
    n = length (GraphBLAS_grbcov) ;
    if (nargout == 0)
        fprintf ('test coverage: %d of %d (%0.4f%%)\n', ...
            covered, n, 100 * (covered / n)) ;
    end
end

