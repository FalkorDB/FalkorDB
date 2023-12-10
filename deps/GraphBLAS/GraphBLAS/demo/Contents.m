% SuiteSparse/GraphBLAS/GraphBLAS/demo: demos for GraphBLAS @GrB interface
%
%   dnn_builtin     - Sparse deep neural network without @GrB
%   gbdemo          - run the graphblas_demo.m
%   gbdemo2         - Extreme performance differences: GraphBLAS vs built-in methods
%   graphblas_demo  - GraphBLAS: graph algorithms in the language of linear algebra
%   graphblas_demo2 - Run the GraphBLAS demo2
%   dnn_builtin2gb  - convert sparse deep neural network from built-in to GraphBLAS
%   dnn_run         - Run the DNN benchmarks
%   mxm_demo        - performance test of real and complex A*B
%   bfs_builtin     - breadth-first-search using purely built-in methods
%   cats_demo       - cats_demo.m
%   demo_nproc      - determine the default # of cores, or set the # of cores to use
%   demo_octave     - return true if Octave is in use, false for MATLAB
%   demo_whoami     - return 'Octave' or 'MATLAB'
%   tmask           - performance tests for logical indexing
%
%   mxm_demo_DellXPS13.txt   - mxm_demo results on Intel Core i7-8565U (4 core)
%   mxm_demo_DGXStation.txt  - mxm_demo results on Intel Xeon E5-2689 (20 core)
%
% The html folder contains the output of graphblas_demo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

