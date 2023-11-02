function gbtest00 (doplots)
%GBTEST00 test GrB.bfs and plot (graph (G))

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    doplots = true ;
end

save_threads = GrB.threads ;
save_chunk   = GrB.chunk ;
GrB.threads (4) ;
GrB.chunk (2) ;

%%MatrixMarket matrix coordinate pattern general
%%GraphBLAS GrB_BOOL
% Matrix from the cover of "Graph Algorithms in the Language of Linear
% Algebra", Kepner and Gilbert.  Note that cover shows A'.  This is A.
% 7 7 12
ij = [
4 1
1 2
4 3
6 3
7 3
1 4
7 4
2 5
7 5
3 6
5 6
2 7 ] ;

source = 1 ;

A = sparse (ij (:,1), ij (:,2), ones (12,1), 8, 8) ;

formats = { 'by row', 'by col' } ;
if (doplots)
    figure (1) ;
    clf ;
end

for k1 = 1:2
    fmt = formats {k1} ;

    A = GrB (A, fmt) ;
    H = GrB (A, 'logical', fmt) ;
    if (k1 == 1 && doplots)
        subplot (1,2,1) ;
        plot (digraph (A)) ;
    end

    v1 = GrB.bfs (H, source) ;
    [v, pi] = GrB.bfs (H, source) ;
    assert (isequal (v, v1)) ;

    vok = [1 2 3 2 3 4 3 0] ;
    assert (isequal (full (double (v)), vok)) ;

    % there are 2 valid trees, and GrB.bfs can return either one
    piok1 = [1 1 4 1 2 3 2 0] ;
    piok2 = [1 1 4 1 2 5 2 0] ;
    ok1 = isequal (full (double (pi)), piok1) ;
    ok2 = isequal (full (double (pi)), piok2) ;
    if (ok1)
        % this tree is more commonly found
        % fprintf ('.') ;
    end
    if (ok2)
        % fprintf ('#') ;
    end
    assert (ok1 || ok2) ;

    G = digraph (H) ;
    v2 = bfsearch (G, source) ;

    levels = full (double (v (v2))) ;
    assert (isequal (levels, sort (levels))) ;

    [v, pi] = GrB.bfs (H, source, 'directed') ;
    assert (isequal (full (double (v)), vok)) ;

    ok1 = isequal (full (double (pi)), piok1) ;
    ok2 = isequal (full (double (pi)), piok2) ;
    if (ok1)
        % this tree is more commonly found
        % fprintf ('+') ;
    end
    if (ok2)
        % this is also valid
        % fprintf ('-') ;
    end
    assert (ok1 || ok2) ;

    [v, pi] = GrB.bfs (H, source, 'directed', 'check') ;
    assert (isequal (full (double (v)), vok)) ;

    ok1 = isequal (full (double (pi)), piok1) ;
    ok2 = isequal (full (double (pi)), piok2) ;
    if (ok1)
        % this tree is more commonly found
        % fprintf ('\\') ;
    end
    if (ok2)
        % this is also valid
        % fprintf ('/') ;
    end
    assert (ok1 || ok2) ;

end

A = A+A' ;
[v, pi] = GrB.bfs (A, 2, 'undirected') ;
if (doplots)
    subplot (1,2,2) ;
    plot (graph (A))
end
vok = [2 1 3 3 2 3 2 0] ;
assert (isequal (full (double (v)), vok)) ;
% two valid trees:
piok1 = [2 2 7 1 2 5 2 0] ;
piok2 = [2 2 7 7 2 5 2 0] ;

    ok1 = isequal (full (double (pi)), piok1) ;
    ok2 = isequal (full (double (pi)), piok2) ;
    if (ok1)
        % this tree is more commonly found
        % fprintf ('@') ;
    end
    if (ok2)
        % fprintf ('_') ;
    end
    assert (ok1 || ok2) ;

GrB.threads (save_threads) ;
GrB.chunk (save_chunk) ;

fprintf ('gbtest00: all tests passed\n') ;


