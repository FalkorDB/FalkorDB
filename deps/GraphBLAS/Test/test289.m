function test289
%TEST289 test the Container for all sparsity formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test289 ----------- C = A using load/unload of a Container\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;
GB_mex_burble (0) ;

rng ('default') ;

% empty hypersparse test
B = GB_spec_random (10, 10, 0, 128, 'double') ;
B.sparsity = 1 ;
C = GB_mex_container (B) ;
GB_spec_compare (B, C) ;

for k1 = 1:length (types)
    atype = types {k1} ;
    fprintf ('\n%s', atype) ;
    for d = [0.5 inf]
        fprintf (' ') ;
        % matrix case
        A = GB_spec_random (10, 10, d, 128, atype) ;
        for A_sparsity = 0:15
            fprintf ('.') ;
            A.sparsity = A_sparsity ;
            C = GB_mex_container (A) ;
            GB_spec_compare (A, C) ;
        end
        fprintf (':') ;
        % emtpy hypersparse case
        B = GB_spec_random (10, 10, 0, 128, atype) ;
        B.sparsity = 1 ;
        C = GB_mex_container (B) ;
        GB_spec_compare (B, C) ;
        fprintf (':') ;
        % vector case
        V = GB_spec_random (10, 1, d, 128, atype) ;
        for V_sparsity = 0:15
            fprintf ('.') ;
            V.sparsity = V_sparsity ;
            C = GB_mex_container (V) ;
            GB_spec_compare (V, C) ;
        end
    end
end

% iso test
clear A
A.matrix = pi * spones (sprand (10, 10, 0.2)) ;
A.iso = true ;
C = GB_mex_container (A) ;
GB_spec_compare (A, C) ;

fprintf ('\n') ;
GB_mex_burble (0) ;
fprintf ('\ntest289: all tests passed\n') ;

