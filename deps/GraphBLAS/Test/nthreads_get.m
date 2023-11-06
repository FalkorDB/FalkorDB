function [nthreads chunk] = nthreads_get
%NTHREADS_GET get # of threads and chunk to use in GraphBLAS
%
% [nthreads chunk] = nthreads_get

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

global GraphBLAS_nthreads
if (isempty (GraphBLAS_nthreads))
    nthreads_set (1) ;
end
nthreads = GraphBLAS_nthreads ;

if (nargout > 1)
    global GraphBLAS_chunk
    if (isempty (GraphBLAS_chunk))
        [nthreads chunk] = nthreads_set (nthreads, 64*1024) ;
    end
    chunk = GraphBLAS_chunk ;
end

