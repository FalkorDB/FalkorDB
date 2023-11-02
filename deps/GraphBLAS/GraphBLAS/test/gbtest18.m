function gbtest18
%GBTEST18 test comparators (and, or, >, ...)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

have_octave = gb_octave ;
tol = 1e-14 ;
rng ('default') ;
for trial = 1:21

    fprintf ('.') ;

    if (mod (trial, 10) == 1)
        m = 1 ;
        n = 1 ;
    else
        m = 4 ;
        n = 5 ;
    end

    MA = sprand (m,n, 0.5) ;    A (2,2) = 2 ; %#ok<*NASGU>
    MB = sprand (m,n, 0.5) ;    B (2,2) = 2 ;

    if (rand < 0.1)
        MA = logical (MA) ;
        MB = logical (MB) ;
    end

    GA = GrB (MA) ;
    GB = GrB (MB) ;

    C1 = (MA < MB) ;
    C2 = (GA < GB) ;
    C3 = (MA < GB) ;
    C4 = (GA < MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA <= MB) ;
    C2 = (GA <= GB) ;
    C3 = (MA <= GB) ;
    C4 = (GA <= MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA > MB) ;
    C2 = (GA > GB) ;
    C3 = (MA > GB) ;
    C4 = (GA > MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA >= MB) ;
    C2 = (GA >= GB) ;
    C3 = (MA >= GB) ;
    C4 = (GA >= MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA == MB) ;
    C2 = (GA == GB) ;
    C3 = (MA == GB) ;
    C4 = (GA == MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA ~= MB) ;
    C2 = (GA ~= GB) ;
    C3 = (MA ~= GB) ;
    C4 = (GA ~= MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    if (~have_octave)
        % octave7: sparse(0).^sparse(1) is wrong; it gives 1 but should be 0.
        if (~islogical (MA))
            C1 = (MA .^ MB) ;
            C2 = (GA .^ GB) ;
            C3 = (MA .^ GB) ;
            C4 = (GA .^ MB) ;
            assert (gbtest_err (C1, C2) < tol) ;
            assert (gbtest_err (C1, C3) < tol) ;
            assert (gbtest_err (C1, C4) < tol) ;
        end
    end

    C1 = (MA & MB) ;
    C2 = (GA & GB) ;
    C3 = (MA & GB) ;
    C4 = (GA & MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;
    if (islogical (MA))
        % C1 = min (MA , MB) ;
        C2 = min (GA , GB) ;
        C3 = min (MA , GB) ;
        C4 = min (GA , MB) ;
        assert (gbtest_eq (C1, C2)) ;
        assert (gbtest_eq (C1, C3)) ;
        assert (gbtest_eq (C1, C4)) ;
    end

    C1 = (MA | MB) ;
    C2 = (GA | GB) ;
    C3 = (MA | GB) ;
    C4 = (GA | MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;
    if (islogical (MA))
        % C1 = max (MA , MB) ;
        C2 = max (GA , GB) ;
        C3 = max (MA , GB) ;
        C4 = max (GA , MB) ;
        assert (gbtest_eq (C1, C2)) ;
        assert (gbtest_eq (C1, C3)) ;
        assert (gbtest_eq (C1, C4)) ;
    end

    C1 = xor (MA , MB) ;
    C2 = xor (GA , GB) ;
    C3 = xor (MA , GB) ;
    C4 = xor (GA , MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = ~MA ;
    C2 = ~GA ;
    assert (gbtest_eq (C1, C2)) ;

    b = (trial - 5) / 10 ;
    if (islogical (MA))
        b = logical (b) ;
    end
    g = GrB (b) ;

    assert (gbtest_eq (MA <  b, GA <  b)) ;
    assert (gbtest_eq (MA <= b, GA <= b)) ;
    % MA > b
    % GA > b
    assert (gbtest_eq (MA >  b, GA >  b)) ;
    assert (gbtest_eq (MA >= b, GA >= b)) ;
    assert (gbtest_eq (MA == b, GA == b)) ;
    assert (gbtest_eq (MA ~= b, GA ~= b)) ;
    if (~islogical (MA))
        MC = MA .^ b ;
        GC = GA .^ b ;
        assert (gbtest_err (MC, GC) < tol) ;
    end

    assert (gbtest_eq (MA <  b, GA <  g)) ;
    assert (gbtest_eq (MA <= b, GA <= g)) ;
    assert (gbtest_eq (MA >  b, GA >  g)) ;
    assert (gbtest_eq (MA >= b, GA >= g)) ;
    assert (gbtest_eq (MA == b, GA == g)) ;
    assert (gbtest_eq (MA ~= b, GA ~= g)) ;
    if (~islogical (MA))
        assert (gbtest_err (MA .^ b, GA .^ g) < tol) ;
    end

    assert (gbtest_eq (MA <  b, MA <  g)) ;
    assert (gbtest_eq (MA <= b, MA <= g)) ;
    assert (gbtest_eq (MA >  b, MA >  g)) ;
    assert (gbtest_eq (MA >= b, MA >= g)) ;
    assert (gbtest_eq (MA == b, MA == g)) ;
    assert (gbtest_eq (MA ~= b, MA ~= g)) ;
    if (~islogical (MA))
        assert (gbtest_err (MA .^ b, MA .^ g) < tol) ;
    end

    assert (gbtest_eq (b <  MA, b <  GA)) ;
    assert (gbtest_eq (b <= MA, b <= GA)) ;
    assert (gbtest_eq (b >  MA, b >  GA)) ;
    assert (gbtest_eq (b >= MA, b >= GA)) ;
    assert (gbtest_eq (b == MA, b == GA)) ;
    assert (gbtest_eq (b ~= MA, b ~= GA)) ;
    if (~islogical (MA))
        assert (gbtest_err (b .^ MA, b .^ GA) < tol) ;
    end

    assert (gbtest_eq (b <  MA, g <  GA)) ;
    assert (gbtest_eq (b <= MA, g <= GA)) ;
    assert (gbtest_eq (b >  MA, g >  GA)) ;
    assert (gbtest_eq (b >= MA, g >= GA)) ;
    assert (gbtest_eq (b == MA, g == GA)) ;
    assert (gbtest_eq (b ~= MA, g ~= GA)) ;
    if (~islogical (MA))
        assert (gbtest_err (b .^ MA, g .^ GA) < tol) ;
    end

    assert (gbtest_eq (b <  MA, g <  MA)) ;
    assert (gbtest_eq (b <= MA, g <= MA)) ;
    assert (gbtest_eq (b >  MA, g >  MA)) ;
    assert (gbtest_eq (b >= MA, g >= MA)) ;
    assert (gbtest_eq (b == MA, g == MA)) ;
    assert (gbtest_eq (b ~= MA, g ~= MA)) ;
    if (~islogical (MA))
        assert (gbtest_err (b .^ MA, g .^ MA) < tol) ;
    end

    k = (mod (trial,2) == 0) ;
    gbk = GrB (k) ;

    C1 = (MA & k) ;
    C2 = (GA & gbk) ;
    C3 = (MA & gbk) ;
    C4 = (GA & k) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;
    if (islogical (MA))
        % C1 = min (MA , k) ;
        C2 = min (GA , gbk) ;
        C3 = min (MA , gbk) ;
        C4 = min (GA , k) ;
        assert (gbtest_eq (C1, C2)) ;
        assert (gbtest_eq (C1, C3)) ;
        assert (gbtest_eq (C1, C4)) ;
    end

    C1 = (k   & MA) ;
    C2 = (gbk & GA) ;
    C3 = (gbk & MA) ;
    C4 = (k   & GA) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA | k) ;
    C2 = (GA | gbk) ;
    C3 = (MA | gbk) ;
    C4 = (GA | k) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = xor (MA, k) ;
    C2 = xor (GA, gbk) ;
    C3 = xor (MA, gbk) ;
    C4 = xor (GA, k) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (k   | MA) ;
    C2 = (gbk | GA) ;
    C3 = (gbk | MA) ;
    C4 = (k   | GA) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = xor (k   , MA) ;
    C2 = xor (gbk , GA) ;
    C3 = xor (gbk , MA) ;
    C4 = xor (k   , GA) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    k = double (k) ;
    gbk = GrB (k) ;
    if (~islogical (MA))
        C1 = (k   .^ MA) ;
        C2 = (gbk .^ GA) ;
        C3 = (gbk .^ MA) ;
        C4 = (k   .^ GA) ;
        assert (gbtest_err (C1, C2) < tol) ;
        assert (gbtest_err (C1, C3) < tol) ;
        assert (gbtest_err (C1, C4) < tol) ;
    end

end

fprintf ('\ngbtest18: all tests passed\n') ;

