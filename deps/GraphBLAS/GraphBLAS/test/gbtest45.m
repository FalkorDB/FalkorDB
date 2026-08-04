function gbtest45 (ghb)
%GBTEST45 test [GrB,GhB].vreduce

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

d.kind = 'sparse' ;
desc = struct ;

for trial = 1:40

    A = rand (4) ;
    G = gtb (ghb, A) ;
    x = gtb_vreduce (ghb, '+', A) ;
    y = gtb_vreduce (ghb, '+', G) ;
    t = gtb_vreduce (ghb, '+', G, d) ;
    z = sum (G, 2) ;
    w = sum (A, 2) ;

    assert (isequal (w, x)) ;
    assert (isequal (w, y)) ;
    assert (isequal (w, z)) ;
    assert (isequal (w, t)) ;

    assert (isequal (class (t), 'double')) ;

    cin = rand (4,1) ;
    x = gtb_vreduce (ghb, cin, '+', '+', A) ;
    y = cin + sum (A, 2) ;
    assert (isequal (x, y)) ;

    m = logical (sprand (4, 1, 0.5)) ;
    x = gtb_vreduce (ghb, cin, m, '+', '+', A) ;
    t = cin + sum (A, 2) ;
    y = cin ;
    y (m) = t (m) ;
    assert (isequal (x, y)) ;

    x = gtb_vreduce (ghb, cin, m, '+', A) ;
    t = sum (A, 2) ;
    y = cin ;
    y (m) = t (m) ;
    assert (isequal (x, y)) ;

    % test internal wrapper
    x = gzb_vreduce (ghb, GrB (cin), GrB (m), GrB (A), '+', desc) ;
    assert (isequal (x, y)) ;

    x = gtb_vreduce (ghb, A, '+') ;
    y = sum (A, 2) ;
    assert (isequal (x, y)) ;
    x = gzb_vreduce (ghb, G, '+') ;
    assert (isequal (x, y)) ;
    x = gzb_vreduce (ghb, G, '+', desc) ;
    assert (isequal (x, y)) ;

end

fprintf ('gbtest45 (%d): all tests passed\n', ghb) ;

