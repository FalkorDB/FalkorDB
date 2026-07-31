function gbtest51 (ghb)
%GBTEST51 test [GrB,GhB].tricount and concatenate

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

files =  {
'./matrix/2blocks'
'./matrix/ash219'
'./matrix/bcsstk01'
'./matrix/bcsstk16'
'./matrix/eye3'
'./matrix/fs_183_1'
'./matrix/ibm32a'
'./matrix/ibm32b'
'./matrix/lp_afiro'
'./matrix/mbeacxc'
'./matrix/t1'
'./matrix/t2'
'./matrix/west0067' } ;
nfiles = length (files) ;

% the files in ./matrix that do not have a .mtx filename are zero-based.
desc.base = 'zero-based' ;

valid_count = [
           0
           0
         160
     1512964
           0
         863
           0
           0
           0
           0
           2
           0
         120 ] ;

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ; %#ok<*ASGLU>

for k = 1:nfiles
    % fprintf ('--------------------------load file:\n') ;
    filename = files {k} ;
    T = load ('-ascii', fullfile (filepath, filename)) ;
    nz = size (T, 1) ;
    X = ones (nz,1) ;
    G = gtb_build (ghb, int64 (T (:,1)), int64 (T (:,2)), X, desc) ;
    A = sparse (T (:,1)+1, T (:,2)+1, X) ;
    assert (isequal (A,G))

    % fprintf ('--------------------------construct G:\n') ;
    [m, n] = size (G) ;
    if (m ~= n)
        G = [gtb(ghb,m,m) G ; G' gtb(ghb,n,n)] ; %#ok<*AGROW>   (concatenate)
    elseif (~issymmetric (G))
        G = G + G' ;
    end

    % fprintf ('--------------------------tricount (G):\n') ;
    c = gtb_tricount (ghb, G) ;
    % fprintf ('triangle count: %-30s : # triangles %d\n', filename, c) ;
    assert (c == valid_count (k)) ;

    % fprintf ('--------------------------convert G to by-row:\n') ;
    G = gtb (ghb, G, 'by row') ;

    % fprintf ('--------------------------tricount (G):\n') ;
    c = gtb_tricount (ghb, G) ;
    assert (c == valid_count (k)) ;
end

% fprintf ('--------------------------tricount (G, ''check''):\n') ;
c = gtb_tricount (ghb, G, 'check') ;
assert (c == valid_count (end)) ;

fprintf ('\ngbtest51 (%d): all tests passed\n', ghb) ;

