function gbtest33 (ghb)
%GBTEST33 test spones, numel, nzmax, size, length, is*, ...
% isempty, issparse, ismatrix, isvector, isscalar, isnumeric,
% isfloat, isreal, isinteger, islogical, isa.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

fprintf ('gbtest33:\n') ;

types = gbtest_types ;

for k1 = 1:length(types)
    type = types {k1} ;
    fprintf ('%s ', type) ;

    H = gtb (ghb, 2^55, 2^55, type) ;
    [m, n] = size (H) ;
    assert (m == 2^55) ;
    assert (n == 2^55) ;
    assert (isequal (class (m), 'int64'))
    assert (isequal (class (n), 'int64'))
    s = size (H) ;
    assert (isequal (s, [2^55 2^55])) ;
    assert (isequal (class (s), 'int64'))

    for k2 = 1:length(types)
        type2 = types {k2} ;

        for n = 0:3
            for m = 0:3
                A = 100 * rand (m, n) ;
                A (A < 50) = 0 ;
                S = sparse (A) ;

                G = gtb (ghb, S, type) ;
                G2 = spones (G, type2) ;
                assert (isequal (gtb_type (ghb, G2), type2)) ;

                C = double (G2) ;
                assert (isequal (sparse (C), spones (S))) ;

                assert (numel (G) == m*n) ;
                e1 = nzmax (G) ;
                e2 = nnz (G) ;
                % G
                % fprintf ('nzmax(G) is %g\n', e1) ;
                % fprintf ('nnz (G)  is %g\n', e2) ;
                assert (e1 >= max (e2, 1))
                assert (isequal (size (G), [m n])) ;
                [m1, n1]  = size (G) ;
                assert (isequal ([m1 n1], [m n])) ;
                if (m == 0 || n == 0)
                    assert (isempty (G)) ;
                else
                    assert (length (G) == max (m, n)) ;
                end
                assert (isempty (G) == (m == 0 | n == 0)) ;
                assert (issparse (G)) ;
                assert (issparse (full (G))) ;
                assert (ismatrix (G)) ;
                assert (isnumeric (G)) ;
                assert (isvector (G) == (m == 1 | n == 1)) ;
                assert (isscalar (G) == (m == 1 & n == 1)) ;

                isfl = gb_contains (type, 'double') | ...
                       gb_contains (type, 'single') ;
                assert (isfloat (G) == isfl) ;
                assert (isreal (G) == (~gb_contains (type, 'complex'))) ;
                isint = isequal (type (1:3), 'int') | ...
                        isequal (type (2:4), 'int') ;
                assert (isinteger (G) == isint) ;
                islog = isequal (type, 'logical') ;
                assert (islogical (G) == islog) ;
                assert (gbtest_isa (ghb, G))
                assert (isa (G, 'numeric')) ;
                assert (isa (G, 'float') == isfl) ;
                assert (isa (G, 'integer') == isint) ;
                assert (isa (G, 'logical') == islog) ;
                assert (isa (G, type) == isequal (gtb_type (ghb, G), type)) ;

                if (ghb == 0)
                    % a GrB object reports true for just "isa GrB"
                    assert (isa (G, 'GrB')) ;
                    assert (~isa (G, 'GhB')) ;
                elseif (ghb == 1)
                    % a GhB object reports true for both queries:
                    assert (isa (G, 'GrB')) ;
                    assert (isa (G, 'GhB')) ;
                elseif (isa (G, 'GhB'))
                    assert (isa (G, 'GrB')) ;
                end
            end
        end
    end
end

fprintf ('\ngbtest33 (%d): all tests passed\n', ghb) ;

