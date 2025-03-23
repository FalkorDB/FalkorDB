function [status] = jit (status)
%GRB.JIT controls the GraphBLAS JIT
%
%   status = GrB.jit ;      % get the current status of the JIT
%   status = GrB.jit (status) ;  % control the JIT and get its status
%
% The GraphBLAS JIT allows GraphBLAS to compile new kernels at run-time
% that are specifically tuned for the particular operators, types, and
% matrix formats.  Without the JIT, only a selected combination of these
% options are computed with high-performance kernels.
%
% GrB.jit controls the GraphBLAS JIT.  Its input/ouput status is a string:
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
% Example:
%
%   [status] = GrB.jit
%   [status] = GrB.jit ('on')
%
% See also GrB.threads, GrB.clear.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    [status] = gbjit ;
elseif (nargin == 1)
    [status] = gbjit (status) ;
end

