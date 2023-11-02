function A = GB_spec_random (m, n, d, scale, type, is_csc,is_hyper,hyper_switch)
%GB_SPEC_RANDOM generate random matrix
%
% A = GB_spec_random (m, n, d, scale, type, is_csc, is_hyper, hyper_switch)
%
% m,n,d: parameters to sprandn (m,n,d)
% m,n: defaults to 4
% d: defaults to 0.5.  If d = inf, A is fully populated.
% scale: a double scalar, defaults to 1.0
% type: a string; defaults to 'double'
% is_csc: true for CSC, false for CSR; defaults to true
% is_hyper: false for non-hypersparse, true for hypersparse, default false

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    m = 4 ;
end

if (nargin < 2)
    n = 4 ;
end

if (nargin < 3)
    d = 0.5 ;
end

if (nargin < 4)
    scale = 1 ;
end

if (nargin < 5)
    type = 'double' ;
end

if (nargin >= 6)
    A.is_csc = is_csc ;
end

if (nargin >= 7 && ~isempty (is_hyper))
    A.is_hyper = is_hyper ;
end

if (nargin >= 8)
    A.hyper_switch = hyper_switch ;
end

if (isinf (d))
    A.matrix = scale * sparse (rand (m, n)) ;
else
    A.matrix = scale * sprandn (m, n, d) ;
end

if (test_contains (type, 'complex'))
    if (isinf (d))
        A.matrix = A.matrix + 1i * scale * sparse (rand (m, n)) ;
    else
        A.matrix = A.matrix + 1i * scale * sprandn (m, n, d) ;
    end
end

if (type (1) == 'u')
    % A is uint8, uint16, uint32, or uint64
    A.matrix = abs (A.matrix) ;
end

A.class = type ;
A.pattern = logical (spones (A.matrix)) ;
A.iso = false ;

