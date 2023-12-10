function ncores = feature_numcores
%FEATURE_NUMCORES determine # of cores the system has

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;
if (have_octave)
    ncores = nproc ;
else
    ncores = feature ('numcores') ;
end

