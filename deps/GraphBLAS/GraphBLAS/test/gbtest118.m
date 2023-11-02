function gbtest118
%GBTEST118 test GrB.argsort

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

m = 20 ;
n = 30 ;
types = gbtest_types ;

for k = 1:length (types)

    type = types {k} ;
    if (gb_contains (type, 'complex'))
        break ;
    end

    fprintf ('%s ', type) ;

    switch (type)
        case 'double'
            range = [-pi pi] ;
        case 'single'
            range = single ([-pi pi]) ;
        case 'logical'
            range = logical ([false true]) ;
        case 'int8'
            range = int8 ([-100 100]) ;
        case 'int16'
            range = int16 ([-100 100]) ;
        case 'int32'
            range = int32 ([-100 100]) ;
        case 'int64'
            range = int32 ([-100 100]) ;
        case 'uint8'
            range = uint8 ([0 100]) ;
        case 'uint16'
            range = uint16 ([0 100]) ;
        case 'uint32'
            range = uint32 ([0 100]) ;
        case 'uint64'
            range = uint32 ([0 100]) ;
        otherwise
            error ('type not supported') ;
    end 

for d = [0.2 inf]

    A = GrB.random (m, n, d, 'range', range) ;

    [C, P] = GrB.argsort (A) ;
    for k = 1:n
        Ak = A (:,k) ;
        [Ai, ~, Ax] = GrB.extracttuples (Ak) ;
        [x, p] = sort (Ax) ;
        nz = length (x) ;
        Ck = C (:,k) ;
        Pk = P (:,k) ;
        Ck = Ck (1:nz) ;
        assert (isequal (Ck, x)) ;
        Pk = double (Pk (1:nz)) ;
        a = Ai (p) ;
        assert (isequal (Pk, a)) ;
    end

    C2 = GrB.argsort (A) ;
    assert (isequal (C, C2)) ;

    [C, P] = GrB.argsort (A, 'descend') ;
    for k = 1:n
        Ak = A (:,k) ;
        [Ai, ~, Ax] = GrB.extracttuples (Ak) ;
        [x, p] = sort (Ax, 'descend') ;
        nz = length (x) ;
        Ck = C (:,k) ;
        Pk = P (:,k) ;
        Ck = Ck (1:nz) ;
        assert (isequal (Ck, x)) ;
        Pk = double (Pk (1:nz)) ;
        a = Ai (p) ;
        assert (isequal (Pk, a)) ;
    end

    [C, P] = GrB.argsort (A, 2) ;
    for k = 1:m
        Ak = A (k,:) ;
        [~, Aj, Ax] = GrB.extracttuples (Ak) ;
        [x, p] = sort (Ax) ;
        nz = length (x) ;
        Ck = C (k,:) ;
        Pk = P (k,:) ;
        Ck = Ck (1:nz) ;
        assert (isequal (Ck', x)) ;
        Pk = double (Pk (1:nz)) ;
        a = Aj (p) ;
        assert (isequal (Pk', a)) ;
    end

    [C, P] = GrB.argsort (A, 2, 'descend') ;
    for k = 1:m
        Ak = A (k,:) ;
        [~, Aj, Ax] = GrB.extracttuples (Ak) ;
        [x, p] = sort (Ax, 'descend') ;
        nz = length (x) ;
        Ck = C (k,:) ;
        Pk = P (k,:) ;
        Ck = Ck (1:nz) ;
        assert (isequal (Ck', x)) ;
        Pk = double (Pk (1:nz)) ;
        a = Aj (p) ;
        assert (isequal (Pk', a)) ;
    end
end

for d = [0.8 inf]
    X = GrB.random (100000, 1, d, 'range', range) ;
    [C, P] = GrB.argsort (X) ;
    [Xi, ~, Xx] = GrB.extracttuples (X) ;
    [x, p] = sort (Xx) ;
    nz = length (x) ;
    Ck = C (1:nz) ;
    assert (isequal (Ck, x)) ;
    Pk = double (P (1:nz)) ;
    a = Xi (p) ;
    assert (isequal (Pk, a)) ;
end

end

fprintf ('\ngbtest118: all tests passed\n') ;
