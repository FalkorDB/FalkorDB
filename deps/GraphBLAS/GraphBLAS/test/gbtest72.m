function gbtest72 (ghb)
%GBTEST72 test any-pair semiring

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

dt = struct ('in0', 'transpose') ;
ntrials = 1000 ;

for n = [1 5 10 100 1000]
%   nfound = 0 ;
    for trial = 1:ntrials
        x = gtb_random (ghb, n, 1, 0.1, 'range', uint32 ([1 255])) ;
        y = gtb_random (ghb, n, 1, 0.1, 'range', uint32 ([1 255])) ;
        c1 = x'*y ;

        c3 = gtb_mxm (ghb, '+.*', x, y, dt) ;
        assert (isequal (c1, c3)) ;

        c2 = gtb_mxm (ghb, 'any.pair', x, y, dt) ;

        c1_present = (gtb_entries (ghb, c1) == 1) ;
        c2_present = (c2 == 1) ;
%       if (c1_present)
%           nfound = nfound + 1 ;
%       end
        assert (c1_present == c2_present) ;
        assert (c1_present == c2) ;

        c4 = gtb_mxm (ghb, 'any.oneb', x, y, dt) ;
        assert (isequal (c2, c4)) ;

    end
%   fprintf ('n: %4d trials: %4d found: %4d\n', n, ntrials, nfound) ;
end

fprintf ('gbtest72 (%d): all tests passed\n', ghb) ;

