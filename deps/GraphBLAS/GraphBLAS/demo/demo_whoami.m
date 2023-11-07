function result = demo_whoami
%DEMO_WHOAMI return 'Octave' or 'MATLAB'

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (demo_octave)
    result = 'Octave' ;
else
    result = 'MATLAB' ;
end

