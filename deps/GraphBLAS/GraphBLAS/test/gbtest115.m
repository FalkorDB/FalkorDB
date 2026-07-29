function gbtest115 (ghb)
%GBTEST115 test serialize/deserialize

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types = gbtest_types ;
compression_methods = { 'none', 'lz4', 'lz4hc', 'zstd', 'debug' } ;

for k = 1:length(types)
    type = types {k} ;
    A = gtb (ghb, GrB.random (5, 10, 0.4, 'range', [0 10]), type) ;

    % defaults
    blob = gtb_serialize (ghb, A) ;
    B = gtb_deserialize (ghb, blob) ;
    assert (isequal (A, B)) ;

    for k2 = 1:5
        method = compression_methods {k2} ;

        % default level
        blob = gtb_serialize (ghb, A, method) ;
        B = gtb_deserialize (ghb, blob) ;
        assert (isequal (A, B)) ;
        B = gtb_deserialize (ghb, GrB (blob)) ;
        assert (isequal (A, B)) ;
        B = gtb_deserialize (ghb, GhB (blob)) ;
        assert (isequal (A, B)) ;

        if (k2 == 3)
            % levels 0:9 for lz4hc
            for level = 0:9
                blob = gtb_serialize (ghb, A, method, level) ;
                B = gtb_deserialize (ghb, blob) ;
                assert (isequal (A, B)) ;
            end
        elseif (k2 == 4)
            % levels 0:19 for zstd
            for level = 0:19
                blob = gtb_serialize (ghb, A, method, level) ;
                B = gtb_deserialize (ghb, blob) ;
                assert (isequal (A, B)) ;
            end
        end
    end
end

fprintf ('\ngbtest115 (%d): all tests passed\n', ghb) ;

