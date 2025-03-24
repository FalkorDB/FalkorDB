function result = semirings
%GRB.SEMIRINGS list all semirings
% Redundant semirings are not listed.  For example '+.*.logical'
% exists, but it is identical to '|.&.logical'.
%
% Example:
%   GrB.semirings ;         % prints a list, with descriptions
%   list = GrB.semirings ;  % returns the list (nothing printed)
%
% See also GrB.semiringinfo.

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

nsemirings = 0 ;
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

    for k2 = 1:nops
        mult = binops {k2,1} ;
        mult_description = binops {k2,2} ;
        first_mult = true ;
        for k3 = 1:length (types)
            type = types {k3} ;
            ok = false ;
            semiring = [add '.' mult '.' type] ;

            % skip redundant logical semirings
            if (isequal (type, 'logical') && ...
                (ismember (mult, skip_logical) || ...
                ismember (add, skip_logical)))
                continue ;
            end

            try
                ok = gbsemiringinfo (semiring) ;
                nsemirings = nsemirings + 1 ;
                if (nargout > 0)
                    result = [result ; semiring] ; %#ok<AGROW>
                end
            catch
                % this is an error, but it is expected since not all
                % combinations operators and types can be used to construct
                % a valid semiring.
            end

            if (ok && nargout == 0)
                if (first_add)
                    fprintf ('\nmonoid: %s', add) ;
                    if (isempty (add_description))
                        fprintf ('\n\n') ;
                    else
                        fprintf (', where %s\n\n', add_description) ;
                    end
                end
                if (first_mult)
                    fprintf ('   %s.%s.type', add, mult) ;
                    if (isempty (mult_description))
                        fprintf ('\n') ;
                    else
                        fprintf (', where %s\n', mult_description) ;
                    end
                    fprintf ('        types: %s', type) ;
                else
                    fprintf (', %s', type) ;
                end
                first_add = false ;
                first_mult = false ;
            end
        end
        if (~first_mult && nargout == 0)
            fprintf ('\n') ;
        end
    end
end

if (nargout == 0)
    fprintf ('\nTotal number of available semirings: %d\n', nsemirings) ;
end

