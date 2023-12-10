function ncores = demo_nproc (ncores_set)
%DEMO_NPROC determine the default # of cores, or set the # of cores to use
%
% Example:
%   ncores = demo_nproc ;       % return default # of cores
%   ncores = demo_nproc (4) ;   % set # cores to 4 (MATLAB only)
%
% See also nproc (Octave), maxNumCompThreads (MATLAB)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (demo_octave)
    % Octave: return the default # of cores, no way to change # of cores to use
    ncores = nproc ;
else
    if (nargin == 0)
        % MATLAB: use the builtin maxNumCompThreads to restore the default
        maxNumCompThreads ('automatic') ;
    else
        % MATLAB: use maxNumCompThreads to set the # of cores to use
        maxNumCompThreads (ncores_set) ;
    end
    % return # of cores now in use
    ncores = maxNumCompThreads ;
end

