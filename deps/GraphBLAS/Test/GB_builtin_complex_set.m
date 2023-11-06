function builtin_complex = GB_builtin_complex_set (builtin_complex)
%GB_BUILTIN_COMPLEX_SET set a global flag to determine the GrB Complex type 
%
% builtin_complex = GB_builtin_complex_set (builtin_complex)
%
% Sets the GraphBLAS_builtin_complex flag.  If true, then the Complex and
% GxB_FC64 types are identical.  If false, the Complex GrB_Type is allocated as
% a user-defined type.
%
% See also GB_builtin_complex_get.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

global GraphBLAS_builtin_complex
if (nargin > 0)
    GraphBLAS_builtin_complex = logical (builtin_complex) ;
elseif (isempty (GraphBLAS_builtin_complex))
    GraphBLAS_builtin_complex = true ;
end

builtin_complex = GraphBLAS_builtin_complex ;

