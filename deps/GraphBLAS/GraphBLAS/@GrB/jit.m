function [s,path] = jit (s,path)
%GRB.JIT controls the GraphBLAS JIT
%
%   s = GrB.jit ;      % get the current status of the JIT
%   s = GrB.jit (s) ;  % control the JIT and get its status
%
% The GraphBLAS JIT allows GraphBLAS to compile new kernels at run-time
% that are specifically tuned for the particular operators, types, and
% matrix formats.  Without the JIT, only a selected combination of these
% options are computed with high-performance kernels.
%
% GrB.jit controls the GraphBLAS JIT.  Its input/ouput s is a string:
%
%   ''          leave the JIT control unchanged.
%   'off'       do not use the JIT, and free any loaded JIT kernels.
%   'pause'     do not run JIT kernels but keep any already loaded.
%   'run'       run JIT kernels if already loaded; no load/compile.
%   'load'      able to load and run JIT kernels; may not compile.
%   'on'        full JIT: able to compile, load, and run.
%   'flush'     clear all loaded JIT kernels, then turn the JIT on;
%               (the same as GrB.jit ('off') ; GrB.jit ('on')).
%
% Refer to the GraphBLAS User Guide for details (GxB_JIT_C_CONTROL).
%
% A second input/output parameter gives the path to a cache folder where
% GraphBLAS keeps the kernels it compiles for the user.  By default, this
% is ~/.SuiteSparse/GrB8.0.0 for GraphBLAS v8.0.0, with a new cache path
% used % for each future @GrB version.
%
% On Apple Silicon, the MATLAB JIT kernels are compiled as x86 binaries,
% but the pure C installation may compile native Arm64 binaries.  Do not
% mix the two.  In this case, set another cache path for MATLAB using
% this method or using GxB_set in the C interface for your native Arm64
% binaries.  See the User Guide for details.
%
% Example:
%
%   [s,path] = GrB.jit
%   [s,path] = GrB.jit ('on', '/home/me/myothercache')
%
% See also GrB.threads, GrB.clear.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    [s, path] = gbjit ;
elseif (nargin == 1)
    [s, path] = gbjit (s) ;
elseif (nargin == 2)
    [s, path] = gbjit (s, path) ;
end

