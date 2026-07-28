function gbtest32 (ghb)
%GBTEST32 test nonzeros

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

for d = 0:.1:1
    for n = 0:10
        A = sprandn (n, n, d) ;
        X = nonzeros (A) ;
        G = gtb (ghb, A) ;
        Y = nonzeros (G) ;
        assert (isequal (X, Y)) ;
    end
end

fprintf ('gbtest32: all tests passed\n', ghb) ;

