function w = GB_spec_Vector_extract (w, mask, accum, u, I, descriptor)
%GB_SPEC_VECTOR_EXTRACT a mimic of GrB_Vector_extract

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargout > 1 || nargin ~= 6)
    error ('usage: w = GB_spec_Vector_extract (w, mask, accum, u, I, desc)') ;
end

% make sure u is a column vector
if (isstruct (u))
    n = size (u.matrix, 2) ;
else
    n = size (u, 2);
end
if (n ~= 1)
    error ('u must be a vector') ;
end

% GraphBLAS does not allow u to be transposed via the descriptor
if (isfield (descriptor, 'inp0'))
    descriptor = rmfield (descriptor, 'inp0') ;
end

w = GB_spec_Matrix_extract (w, mask, accum, u, I, 1, descriptor) ;

