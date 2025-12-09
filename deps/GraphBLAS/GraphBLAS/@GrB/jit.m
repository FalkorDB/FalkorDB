function [status,path] = jit (status,path)
%GRB.JIT controls the GraphBLAS JIT
%
%   status = GrB.jit ;      % get the current status of the JIT
%   status = GrB.jit (status) ;  % control the JIT and get its status
%   [status,path] = GrB.jit (status,path) ; % get/set JIT cache path
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
% The 2nd input/output parameter is a string that defines the JIT cache
% path.  If you run multiple instances of MATLAB at the same time, each
% must use a different cache folder.  The default cache on Linux/Mac is
% ~/.SuiteSparse/GrB10.1.1 (for GraphBLAS v10.1.1 for example).  On
% Windows, it is located inside your AppData\Local folder.  If you change
% to another location, adding the GraphBLAS version is recommended; see
% the last example below.
%
% Refer to the GraphBLAS User Guide for details (GxB_JIT_C_CONTROL
% and GxB_JIT_CACHE_PATH).
%
% Example:
%
%   [status] = GrB.jit
%   [status] = GrB.jit ('on')
%   [status,path] = GrB.jit
%   [status,path] = GrB.jit ('on', [userpath filesep 'my_other_cache')
%
%   % setting the JIT cache path with the GraphBLAS version included:
%   v = GrB.ver
%   [status,path] = GrB.jit ('on', [userpath filesep 'myGrB' v.Version])
%
% See also GrB.threads, GrB.clear.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)

    if (nargout <= 1)
        [status] = gbjit ;
    else
        [status, path] = gbjit ;
    end

elseif (nargin == 1)

    if (nargout <= 1)
        [status] = gbjit (status) ;
    else
        [status, path] = gbjit (status) ;
    end

else

    if (nargout <= 1)
        [status] = gbjit (status, path) ;
    else
        [status, path] = gbjit (status, path) ;
    end

end

