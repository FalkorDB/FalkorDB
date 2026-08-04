function gbtest47 (ghb)
%GBTEST47 test [GrB,GhB].entries, [GrB,GhB].nonz, numel

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = 100 * rand (4) ;
X = rand (4) ;
types = gbtest_types ;
for k = 1:length (types)
    fprintf ('.') ;
    type = types {k} ;
    if (isequal (type, 'single complex'))
        B = complex (single (A), single (X)) ;
    elseif (isequal (type, 'double complex'))
        B = complex (A, X) ;
    else
        B = gbtest_cast (A, type) ;
    end
    x1 = gtb_entries (ghb, B, 'list') ;
    x2 = unique (nonzeros (B)) ;
    assert (isequal (x1, x2)) ;
    assert (isequal (type, gtb_type (ghb, x1))) ;
    assert (isequal (type, gtb_type (ghb, x2))) ;
end

A = magic (4) ;
c0 = nnz (A) ;
c1 = gtb_nonz (ghb, A) ;
c2 = gtb_nonz (ghb, gtb (ghb, A)) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

c1 = gtb_nonz (ghb, A, 0) ;
c2 = gtb_nonz (ghb, gtb (ghb, A), 0) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

A = sparse (A) ;
c0 = nnz (A) ;

c1 = gtb_nonz (ghb, A) ;
c2 = gtb_nonz (ghb, gtb (ghb, A)) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

c1 = gtb_nonz (ghb, A, 0) ;
c2 = gtb_nonz (ghb, gtb (ghb, A), 0) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

A = sparse (A) ;
c0 = nnz (A ~= 1) ;
c1 = gtb_nonz (ghb, A, 1) ;
c2 = gtb_nonz (ghb, gtb (ghb, A), 1) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

try
    x = vpa (1) ; %#ok<*NASGU>
    have_symbolic = true ;
    fprintf ('\nwith symbolic toolbox\n') ;
catch
    % symbolic toolbox not available
    have_symbolic = false ;
    fprintf ('\nno symbolic toolbox\n') ;
end


for trial = 1:40
    fprintf ('(%d)', trial) ;

    A = rand (4) ;
    A (A > .5) = 0 ;
    G = gtb (ghb, A) ;

    c1 = gtb_entries (ghb, A) ;
    c2 = gtb_entries (ghb, G) ;
    assert (c1 == c2) ;
    assert (c1 == numel (A)) ;

    c1 = gtb_nonz (ghb, A) ;
    c2 = gtb_nonz (ghb, G) ;
    assert (c1 == c2) ;
    assert (c1 == nnz (A)) ;

    B = sparse (A) ;
    G = gtb (ghb, B) ;

    c1 = gtb_entries (ghb, B) ;
    c2 = gtb_entries (ghb, G) ;
    assert (c1 == c2) ;
    assert (c1 == nnz (B)) ;

    c1 = gtb_nonz (ghb, B) ;
    c2 = gtb_nonz (ghb, G) ;
    assert (c1 == c2) ;
    assert (c1 == nnz (B)) ;

    c1 = gtb_nonz (ghb, B, 0) ;
    c2 = gtb_nonz (ghb, G, 0) ;
    assert (c1 == c2) ;
    assert (c1 == nnz (B)) ;

    x1 = gtb_entries (ghb, B, 'list') ;
    x2 = gtb_entries (ghb, G, 'list') ;
    assert (isequal (x1, x2)) ;

    x1 = gtb_nonz (ghb, B, 'list') ;
    x2 = gtb_nonz (ghb, G, 'list') ;
    x0 = unique (nonzeros (B)) ;
    assert (isequal (x0, x2)) ;
    assert (isequal (x1, x2)) ;

    d1 = gtb_entries (ghb, B, 'row') ;
    d2 = gtb_entries (ghb, G, 'row') ;
    d3 = length (find (sum (spones (B), 2))) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gtb_nonz (ghb, B, 'row') ;
    d2 = gtb_nonz (ghb, G, 'row') ;
    d3 = length (find (sum (spones (B), 2))) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gtb_entries (ghb, B, 'row', 'list') ;
    d2 = gtb_entries (ghb, G, 'row', 'list') ;
    d3 = find (sum (spones (B), 2)) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gtb_nonz (ghb, B, 'row', 'list') ;
    d2 = gtb_nonz (ghb, G, 'row', 'list') ;
    d3 = find (sum (spones (B), 2)) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gtb_entries (ghb, B, 'col') ;
    d2 = gtb_entries (ghb, G, 'col') ;
    d3 = length (find (sum (spones (B), 1))) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gtb_nonz (ghb, B, 'col') ;
    d2 = gtb_nonz (ghb, G, 'col') ;
    d3 = length (find (sum (spones (B), 1))) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gtb_entries (ghb, B, 'col', 'list') ;
    d2 = gtb_entries (ghb, G, 'col', 'list') ;
    d3 = find (sum (spones (B), 1))' ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gtb_nonz (ghb, B, 'col', 'list') ;
    d2 = gtb_nonz (ghb, G, 'col', 'list') ;
    d3 = find (sum (spones (B), 1))' ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gtb_nonz (ghb, B, 'col', 'degree') ;
    d2 = gtb_nonz (ghb, G, 'col', 'degree') ;
    d3 = int64 (full (sum (spones (B), 1)))' ;
    assert (isequal (d1, d2)) ;
    assert (isequal (double (d1), sparse (double (d3)))) ;

    fprintf ('[') ;

    % requires vpa in the Symbolic toolbox:
    if (have_symbolic)
        Huge = gtb (ghb, 2^30, 2^30) ;
        e = numel (Huge) ;
        assert (logical (e == 2^60)) ;
    end

    fprintf (']') ;
end

fprintf ('\ngbtest47 (%d): all tests passed\n', ghb) ;

