function C = gb_mexfunction_result (ghb, C_opaque, kind)
%GB_MEXFUNCTION_RESULT result of a GraphBLAS mexFunction.  Not user-callable.
% Returns a matrix the C_opaque handle of a GrB_Matrix as computed by a
% GraphBLAS mexFunction.  The matrix is returned as a GrB or GhB matrix if kind
% is 0, or MATLAB/Octave otherwise.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (kind == 0)
    % return a GrB or GhB object, depending on the ghb parameter
    C = gzb (ghb, C_opaque) ;
else
    % return a built-in MATLAB/Octave matrix from the C_opaque handle
    C = gb_builtin (gzb (ghb, C_opaque)) ;
end

