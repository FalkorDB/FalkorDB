function gbtest25 (ghb)
%GBTEST25 test diag, tril, triu

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;
desc = struct ;

for trials = 1:10
    fprintf ('.') ;

    for m = 2:6
        for n = 2:6
            A = sprand (m, n, 0.5) ;
            G = gtb (ghb, A) ;
            for k = -m:n
                B = diag (A, k) ;
                C = diag (G, k) ;
                assert (gbtest_eq (B, C)) ;
                C = diag (G, gtb (ghb, k)) ;
                assert (gbtest_eq (B, C)) ;
                B = tril (A, k) ;
                C = tril (G, k) ;
                assert (gbtest_eq (B, C)) ;
                B = triu (A, k) ;
                C = triu (G, k) ;
                C2 = gzb_select (ghb, 'triu', G, GrB (k), desc) ;
                assert (gbtest_eq (B, C)) ;
                assert (gbtest_eq (B, C2)) ;
            end
            B = diag (A) ;
            C = diag (G) ;
            assert (gbtest_eq (B, C)) ;
        end
    end

    for m = 1:6
        A = sprandn (m, 1, 0.5) ;
        G = gtb (ghb, A) ;
        for k = -6:6
            B = diag (A, k) ;
            C = diag (G, k) ;
            assert (gbtest_eq (B, C)) ;
            C2 = gzb_mdiag (ghb, GrB (A), k) ;
            assert (gbtest_eq (B, C2)) ;
            B = tril (A, k) ;
            C = tril (G, k) ;
            assert (gbtest_eq (B, C)) ;
            B = triu (A, k) ;
            C = triu (G, k) ;
            assert (gbtest_eq (B, C)) ;
        end

        B = diag (A) ;
        C = diag (G) ;
        assert (gbtest_eq (B, C)) ;
        B = tril (A) ;
        C = tril (G) ;
        assert (gbtest_eq (B, C)) ;
        B = triu (A) ;
        C = triu (G) ;
        assert (gbtest_eq (B, C)) ;
    end
end

n = uint64 (2^60) ;
A = magic (5) ;
I = [1 2 3 4 5] ;
H = gtb (ghb, n,n) ;
H (I,I) = A ;
d = diag (H) ;
[~,~,x] = find (d) ;
e = diag (A) ;
assert (isequal (e, x))

for k = 1:length(I)
    i = I (k) - 1 ;
    d = diag (H, i) ;
    [~,~,x] = find (d) ;
    e = diag (A, k-1) ;
    assert (isequal (e, x)) ;
    d = diag (H, -i) ;
    [~,~,x] = find (d) ;
    e = diag (A, -(k-1)) ;
    assert (isequal (e, x)) ;
end

I = [1 2 3 n-1 n] ;
H = gtb (ghb, n,n) ;
H (I,I) = A ;
d = diag (H, n-2) ;
[~,~,x] = find (d) ;
e = diag (A, 3) ;
assert (isequal (e, x)) ;

fprintf ('\ngbtest25 (%d): all tests passed\n', ghb) ;

