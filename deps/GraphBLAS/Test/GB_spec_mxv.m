function w = GB_spec_mxv (w, mask, accum, semiring, A, u, descriptor)
%GB_SPEC_MXV a mimic of GrB_mxv
%
% Usage:
% w = GB_spec_mxv (w, mask, accum, semiring, A, u, descriptor)
%
% w, mask, and u are vectors.  u is not transposed (descriptor inp1 ignored)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargout > 1 || nargin ~= 7)
    error ('usage: w = GB_spec_mxv (w, mask, accum, semiring, A, u, descriptor)') ;
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
if (isfield (descriptor, 'inp1'))
    descriptor = rmfield (descriptor, 'inp1') ;
end

w = GB_spec_mxm (w, mask, accum, semiring, A, u, descriptor) ;

