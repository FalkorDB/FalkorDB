function w = GB_spec_vxm (w, mask, accum, semiring, u, A, descriptor)
%GB_SPEC_VXM a mimic of GrB_vxm
%
% Usage:
% w = GB_spec_vxm (w, mask, accum, semiring, u, A, descriptor)
%
% w, mask, and u are column vectors.  Computes w'=u'*A or w'=u'*A'

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargout > 1 || nargin ~= 7)
    error ('usage: w = GB_spec_vxm (w, mask, accum, semiring, u, A, descriptor)') ;
end

% In the C implementation of GraphBLAS itself, the column vectors u and w are
% not transposed since it is costly to transpose column vectors.  w=A*u or
% w=A'*u is computed instead, but with the multiply operator fmult(x,y) used as
% fmult(y,x) instead.  This also requires the descriptor to be revised for A.
%
% The transformation of the problem in the C implementation is the same as
% simply doing the transpose of u and w and leaving the descriptor unchanged.
% Then the inputs to the multiply operator are used as-is and not flipped.
% This simpler method is used in this mimic.

% make sure u is a column vector on input, then transpose it
if (isstruct (u))
    n = size (u.matrix, 2) ;
    u.matrix = u.matrix.' ;
    u.pattern = u.pattern' ;
else
    n = size (u, 2);
    u = u' ;
end
if (n ~= 1)
    error ('u must be a vector') ;
end

% make sure w is a column vector on input, then transpose it
if (isstruct (w))
    n = size (w.matrix, 2) ;
    w.matrix = w.matrix.' ;
    w.pattern = w.pattern' ;
else
    n = size (w, 2);
    w = w' ;
end
if (n ~= 1)
    error ('w must be a vector') ;
end

% mask is a column vector on input, so transpose it
mask = mask' ;

% GraphBLAS does not allow u to be transposed via the descriptor
if (isfield (descriptor, 'inp0'))
    descriptor = rmfield (descriptor, 'inp0') ;
end

w = GB_spec_mxm (w, mask, accum, semiring, u, A, descriptor) ;

% transpose w back into a column vector
w.matrix = w.matrix.' ;
w.pattern = w.pattern' ;

