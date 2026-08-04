function gbtest121 (ghb)
%GBTEST121 test times with scalars

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

a = pi ;
b = 2 ;
c1 = a.*b ;
c2 = gtb (ghb, a) .* gtb (ghb, b) ;

assert (isequal (c1, c2)) ;

fprintf ('gbtest121 (%d): all tests passed\n', ghb) ;

