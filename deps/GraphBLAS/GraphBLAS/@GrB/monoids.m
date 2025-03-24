function result = monoids
%GRB.MONOIDS list all monoids
% Redundant monoids are not listed.  For example '+.logical'
% exists, but it is identical to '|.logical'.
%
% Example:
%   GrB.monoids ;           % prints a list, with descriptions
%   list = GrB.monoids ;    % returns the list (nothing printed)
%
% See also GrB.monoidinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

types = {
    'logical'
    'double'
    'single'
    'int8'
    'int16'
    'int32'
    'int64'
    'uint8'
    'uint16'
    'uint32'
    'uint64'
    'single complex'
    'double complex'
    } ;

binops = gb_binops ;

useful_monoids = {'any', 'min', 'max', '+', '*', '==', '~=', '|', '&', ...
    'xor', 'xnor', 'bitand', 'bitor', 'bitxor', 'bitxnor' } ;

skip_logical = {'min', 'max', '+', '-', 'rminus', '*', '/', '\', 'iseq', ...
    'isne', 'isgt', 'islt', 'isle', 'isge', 'pow', '~=', '==' } ;

nmonoids = 0 ;
nops = size (binops, 1) ;

if (nargout > 0)
    result = { } ;
end

for k1 = 1:nops
    add = binops {k1,1} ;
    add_description = binops {k1,2} ;
    first_add = true ;

    % skip redundant monoids
    if (~ismember (add, useful_monoids))
        continue ;
    end

        for k3 = 1:length (types)
            type = types {k3} ;
            ok = false ;
            monoid = [add '.' type] ;

            % skip redundant logical monoids
            if (isequal (type, 'logical') && ...
                ismember (add, skip_logical))
                continue ;
            end

            try
                ok = gbmonoidinfo (monoid) ;
                nmonoids = nmonoids + 1 ;
                if (nargout > 0)
                    result = [result ; monoid] ; %#ok<AGROW>
                end
            catch
                % this is an error, but it is expected since not all
                % combinations operators and types can be used to construct
                % a valid monoid.
            end
            if (ok && nargout == 0)
                if (first_add)
                    fprintf ('\nmonoid: %s', add) ;
                    if (isempty (add_description))
                        fprintf ('\n') ;
                    else
                        fprintf (', where %s\n', add_description) ;
                    end
                    fprintf ('        types: %s', type) ;
                else
                    fprintf (', %s', type) ;
                end
                first_add = false ;
            end
        end
        if (~first_add && nargout == 0)
            fprintf ('\n') ;
        end

end

if (nargout == 0)
    fprintf ('\nTotal number of available monoids: %d\n', nmonoids) ;
end

