function [nthreads chunk] = nthreads_set (nthreads, chunk)
%NTHREADS_SET set # of threads and chunk to use in GraphBLAS
%
% [nthreads chunk] = nthreads_set (nthreads, chunk)
%
% If nthreads is empty, or if no input arguments, nthreads is set to 1.
% If chunk is empty, or if no input arguments, chunk is not modified.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

global GraphBLAS_nthreads
if (nargin < 1)
    nthreads = [ ] ;
end
if (isempty (nthreads))
    nthreads = int32 (1) ;
end
nthreads = int32 (nthreads) ;
GraphBLAS_nthreads = nthreads ;

if (nargin > 1 || nargout > 1)
    global GraphBLAS_chunk
    if (nargin > 1)
        GraphBLAS_chunk = chunk ;
    elseif (isempty (GraphBLAS_chunk))
        GraphBLAS_chunk = 64*1024 ;
    end
    if (nargout > 1)
        chunk = GraphBLAS_chunk ;
    end
end

