function result = gb_octave
%GB_OCTAVE return true if Octave is in use, false for MATLAB

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

result = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;
