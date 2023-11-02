function x = GB_spec_zeros (mn, type)
%GB_SPEC_ZEROS all-zero matrix of a given type.
% x = GB_spec_zeros ([m n], type) returns a dense built-in matrix of all zeros
% with the given type.  The type is a string, 'logical', 'int8', 'int16',
% 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'single', 'double',
% 'single complex', and 'double complex'.
%
% See also GB_spec_type, GB_spec_zeros, zeros.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 2)
    type = 'double' ;
end

if (isequal (type, 'single complex'))
    x = complex (zeros (mn, 'single')) ;
elseif (isequal (type, 'double complex'))
    x = complex (zeros (mn)) ;
else
    x = zeros (mn, type) ;
end


