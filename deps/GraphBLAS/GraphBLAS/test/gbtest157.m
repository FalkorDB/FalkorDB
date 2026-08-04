function gbtest157
%GBTEST157 test GhB.bfs on many matrices (push vs pull)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load (fullfile (filepath, './matrix/west0479_correct.mat')) ;

A = Problem.A ;
deg = GhB.entries (A, 'row', 'degree') ;
AT = logical (spones (A))' ;
n = size (A, 1) ;
G = digraph (A) ;

for k = 1:n

    [v1, parent] = GhB.bfs (A, AT, deg, k, 'parent') ;
    v2 = bfsearch (G, k) ;
    [found1,~] = find (v1) ;
    found2 = sort (v2) ;
    assert (isequal (double (found1), found2)) ;

    [v1, parent] = GhB.bfs (A, AT, deg, k, 'maxparent') ;
    v2 = bfsearch (G, k) ;
    [found1,~] = find (v1) ;
    found2 = sort (v2) ;
    assert (isequal (found1, found2)) ;

end

files = {
    './matrix/bcsstk16',    [ 4734] ,
    './matrix/eye3',        [ ],
    './matrix/fs_183_1',    [ ] } ;

nfiles = size (files, 1) ;
for j = 1:nfiles

    filename = files {j,1} ;
    sources = files {j,2} ;
    fprintf ('file: %s\n', filename) ;
    T = load ('-ascii', fullfile (filepath, filename)) ;
    A = sparse (T (:,1) + 1, T (:,2) + 1, T (:,3)) ;

    [m n] = size (A) ;
    if (m ~= n)
        filename
        error ('drop this') ;
    end

    deg = GhB.entries (A, 'row', 'degree') ;
    AT = logical (spones (A))' ;
    n = size (A, 1) ;
    G = digraph (A) ;

    if (isempty (sources))
        sources = 1:n ;
    end

    for k = sources

        [v1, parent] = GhB.bfs (A, AT, deg, k, 'minparent') ;
        v2 = bfsearch (G, k) ;
        [found1,~] = find (v1) ;
        found2 = sort (v2) ;
        assert (isequal (double (found1), found2)) ;

    end

    A = A+A' ;
    deg = GhB.entries (A, 'row', 'degree') ;
    AT = logical (spones (A))' ;
    G = digraph (A) ;

    for k = sources

        [v1, parent] = GhB.bfs (A, AT, deg, k, 'maxparent', 'undirected') ;
        v2 = bfsearch (G, k) ;
        [found1,~] = find (v1) ;
        found2 = sort (v2) ;
        assert (isequal (double (found1), found2)) ;

    end

end

files = { './matrix/GD00_c.mat', './matrix/GD96_a.mat' } ;

for j = 1:length (files)
    file = files {j} ;
    fprintf ('file: %s\n', file) ;
    load (fullfile (filepath, files {j})) ;
    A = Problem.A ;
    AT = A' ;
    deg = GhB.entries (A, 'row', 'degree') ;
    n = size (A,1) ;
    G = digraph (A) ;
    for k = 1:n
        [v1, parent] = GhB.bfs (A, AT, deg, k, 'minparent') ;
        v2 = bfsearch (G, k) ;
        [found1,~] = find (v1) ;
        found2 = sort (v2) ;
        assert (isequal (double (found1), found2)) ;
    end
end

fprintf ('gbtest157: all tests passed\n') ;

