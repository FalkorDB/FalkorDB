function gbtest115
%GBTEST115 test serialize/deserialize

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

types = gbtest_types ;
compression_methods = { 'none', 'lz4', 'lz4hc', 'zstd', 'debug' } ;

for k = 1:length(types)
    type = types {k} ;
    A = GrB (GrB.random (5, 10, 0.4, 'range', [0 10]), type) ;

    % defaults
    blob = GrB.serialize (A) ;
    B = GrB.deserialize (blob) ;
    assert (isequal (A, B)) ;

    for k2 = 1:5
        method = compression_methods {k2} ;

        % default level
        blob = GrB.serialize (A, method) ;
        B = GrB.deserialize (blob) ;
        assert (isequal (A, B)) ;

        B = GrB.deserialize (blob, 'fast') ;
        assert (isequal (A, B)) ;

        B = GrB.deserialize (blob, 'secure') ;
        assert (isequal (A, B)) ;

        B = GrB.deserialize (blob, 'secure', type) ;
        assert (isequal (A, B)) ;

        if (k2 == 3)
            % levels 0:9 for lz4hc
            for level = 0:9
                blob = GrB.serialize (A, method, level) ;
                B = GrB.deserialize (blob) ;
                assert (isequal (A, B)) ;
            end
        elseif (k2 == 4)
            % levels 0:19 for zstd
            for level = 0:19
                blob = GrB.serialize (A, method, level) ;
                B = GrB.deserialize (blob) ;
                assert (isequal (A, B)) ;
            end
        end
    end
end

fprintf ('\ngbtest115: all tests passed\n') ;


