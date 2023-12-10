function nthreads = threads (nthreads)
%GRB.THREADS get/set the max number of threads to use in GraphBLAS.
%
%   nthreads = GrB.threads ;      % get the current maximum # of threads
%   GrB.threads (nthreads) ;      % set the maximum # of threads
%
% GrB.threads gets and/or sets the maximum number of threads to use in
% GraphBLAS.  By default, if GraphBLAS has been compiled with OpenMP, it
% uses the number of threads returned by omp_get_max_threads.  Otherwise,
% it can only use a single thread.
%
% Changing the number of threads with GrB.threads(nthreads) causes all
% subsequent GraphBLAS operations to use at most the given number of
% threads.  GraphBLAS may use fewer threads, if the problem is small (see
% GrB.chunk).  The setting is kept for the remainder of the current
% session, or until 'clear all' or GrB.clear is used, at which point the
% setting reverts to the default number of threads.
%
% Example:
%
%   GrB.threads (8) ;  % GraphBLAS will use at most 8 threads
%
% See also GrB.chunk.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    nthreads = gbthreads ;
else
    nthreads = gbthreads (nthreads) ;
end

