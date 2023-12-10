function [C,P] = argsort (A, arg1, arg2)
% GRB.ARGSORT sort the rows or columns of a matrix 
%
% [C,P] = argsort (A)
% [C,P] = argsort (A, 'ascend')
% [C,P] = argsort (A, 'descend')
% [C,P] = argsort (A, dim)
% [C,P] = argsort (A, dim, 'ascend')
% [C,P] = argsort (A, dim, 'descend')
%
% GrB.argsort sorts the rows or columns of A.  By default, the
% columns of A are sorted (dim == 1); with dim = 2, the rows of A are
% sorted.  The default is to sort in ascending order.
%
% Example:
%
%   A = sprand (20,10,0.5) ;
%   [C,P] = GrB.argsort (A) ;
%   A (:,1)
%   C (:,1)
%   P (:,1)
%   % the MATLAB sort includes all implicit zeros:
%   [B,I] = sort (A) ;
%   B (:,1)
%   I (:,1)
%
% This methods differs from the MATLAB sort function.  Implicit zeros
% are ignored and always placed last in the output.  P is returned sparse,
% and only reflects the entries in A, not the implicit zeros.  The MATLAB
% [C,P] = sort (A) always returns P as full since it permutes the implicit
% zeros of A as well.  Complex matrices are not supported.
%
% See also sort, GrB.argmin, GrB.argmax.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

dim = 1 ;
direction = 'ascend' ;

if (nargin == 3)
    dim = arg1 ;
    direction = arg2 ;
elseif (nargin == 2)
    if (ischar (arg1))
        direction = arg1 ;
    else
        dim = arg1 ;
    end
end

if (nargout == 1)
    C = gbargsort (A, dim, direction) ;
else
    [C,P] = gbargsort (A, dim, direction) ;
    P = GrB (P) ;
end

C = GrB (C) ;

