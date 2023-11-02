function test108
%TEST108 test boolean monoids

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% only well-defined if op is associative

ops = {
% 10 operators where x,y,z are all the same class
% 'first',     % z = x
% 'second',    % z = y
'min',       % z = min(x,y)
'max',       % z = max(x,y)
'plus',      % z = x + y
'minus',     % z = x - y
'rminus',    % z = y - z
'times',     % z = x * y
% 'div',       % z = x / y
% 'rdiv',      % z = y / x
% 6 comparators where x,y,z are all the same class
'iseq',      % z = (x == y)
'isne',      % z = (x != y)
% 'isgt',      % z = (x >  y)
% 'islt',      % z = (x <  y)
% 'isge',      % z = (x >= y)
% 'isle',      % z = (x <= y)
% 3 boolean operators where x,y,z are all the same class
'or',        % z = x || y
'and',       % z = x && y
'xor'        % z = x != y
%----------------------------
% 6 comparators where x,y are all the same class, z is logical
'eq',        % z = (x == y)
'ne',        % z = (x != y)
% 'gt',        % z = (x >  y)
% 'lt',        % z = (x <  y)
% 'ge',        % z = (x >= y)
% 'le',        % z = (x <= y)
} ;

rng ('default') ;

for d = 0:10

    clear A
    m = 4 ;
    n = 4 ;
    % d = 0.8 ;
    A.matrix = spones (sprand (m, n, d/10)) ;
    A.class = 'logical' ;
    nz = nnz (A.matrix) ;
    e = rand (nz, 1) ;
    if (d == 3)
        A.values = true (nz,1) ;
    elseif (d == 4)
        A.values = false (nz,1) ;
    else
        A.values = (e > 0.5) ;
    end
    X = A.values ;
    n = length (X) ;

%   fprintf ('\n============================================== %d: %d\n', d, nz) ;

    nops = length (ops) ;
    Results = nan (nops, 2, 2, 2) ;

    for k = 1:length (ops)
        op = ops {k} ;
%       fprintf ('\n============================== %s\n', op) ;
        for first = 0:1
            for last = 0:1
                A.values (1) = first ;
                A.values (end) = last ;
                X = A.values ;
                for id = 0:1

                    identity = logical (id) ;

                    if (isequal (op, 'or'))
                        % with terminal and typecasting
                        A.class = 'single' ;
                        result = GB_mex_reduce_bool (A, op, identity, false) ;
                        A.class = 'logical' ;
                    else
                        % no terminal
                        result = GB_mex_reduce_bool (A, op, identity) ;
                    end

                    % now compute with built-in methods

                    % known identity values
                    z = identity ;
                    % 8 operators where x,y,z are all the same class
                    if (isequal (op, 'first'))    z = identity      ; end
                    if (isequal (op, 'second'))   z = identity      ; end
                    if (isequal (op, 'min'))      z = true          ; end   % and
                    if (isequal (op, 'max'))      z = false         ; end   % or
                    if (isequal (op, 'plus'))     z = false         ; end   % or
                    if (isequal (op, 'minus'))    z = false         ; end   % xor
                    if (isequal (op, 'rminus'))   z = false         ; end   % xor
                    if (isequal (op, 'times'))    z = true          ; end   % and
                    if (isequal (op, 'div'))      z = identity      ; end   % first
                    if (isequal (op, 'rdiv'))     z = identity      ; end   % second
                    % 6 comparators where x,y,z are all the same class
                    if (isequal (op, 'iseq'))     z = true          ; end   % eq
                    if (isequal (op, 'isne'))     z = false         ; end   % xor
                    if (isequal (op, 'isgt'))     z = identity      ; end   % gt
                    if (isequal (op, 'islt'))     z = identity      ; end   % lt
                    if (isequal (op, 'isge'))     z = identity      ; end   % ge
                    if (isequal (op, 'isle'))     z = identity      ; end   % le
                    % 3 boolean operators where x,y,z are all the same class
                    if (isequal (op, 'or'))       z = false         ; end
                    if (isequal (op, 'and'))      z = true          ; end
                    if (isequal (op, 'xor'))      z = false         ; end
                    %----------------------------
                    % 6 comparators where x,y are all the same class, z is logical
                    if (isequal (op, 'eq'))       z = true          ; end
                    if (isequal (op, 'ne'))       z = false         ; end   % xor
                    if (isequal (op, 'gt'))       z = identity      ; end
                    if (isequal (op, 'lt'))       z = identity      ; end
                    if (isequal (op, 'ge'))       z = identity      ; end
                    if (isequal (op, 'le'))       z = identity      ; end

                    for i = 1:n
                        x = z ;
                        y = X (i) ;

                        % 8 operators where x,y,z are all the same class
                        if (isequal (op, 'first'))    z = x             ; end
                        if (isequal (op, 'second'))   z = y             ; end
                        if (isequal (op, 'min'))      z = min(x,y)      ; end
                        if (isequal (op, 'max'))      z = max(x,y)      ; end
                        if (isequal (op, 'plus'))     z = or (x,y)      ; end
                        if (isequal (op, 'minus'))    z = xor(x,y)      ; end
                        if (isequal (op, 'rminus'))   z = xor(x,y)      ; end
                        if (isequal (op, 'times'))    z = x * y         ; end
                        if (isequal (op, 'div'))      z = x             ; end   % boolean division == first
                        if (isequal (op, 'rdiv'))     z = y             ; end   % boolean division == second
                        % 6 comparators where x,y,z are all the same class
                        if (isequal (op, 'iseq'))     z = (x == y)      ; end
                        if (isequal (op, 'isne'))     z = (x ~= y)      ; end
                        if (isequal (op, 'isgt'))     z = (x >  y)      ; end
                        if (isequal (op, 'islt'))     z = (x <  y)      ; end
                        if (isequal (op, 'isge'))     z = (x >= y)      ; end
                        if (isequal (op, 'isle'))     z = (x <= y)      ; end
                        % 3 boolean operators where x,y,z are all the same class
                        if (isequal (op, 'or'))       z = x || y        ; end
                        if (isequal (op, 'and'))      z = x && y        ; end
                        if (isequal (op, 'xor'))      z = xor(x,y)      ; end
                        %----------------------------
                        % 6 comparators where x,y are all the same class, z is logical
                        if (isequal (op, 'eq'))       z = (x == y)      ; end
                        if (isequal (op, 'ne'))       z = (x ~= y)      ; end
                        if (isequal (op, 'gt'))       z = (x >  y)      ; end
                        if (isequal (op, 'lt'))       z = (x <  y)      ; end
                        if (isequal (op, 'ge'))       z = (x >= y)      ; end
                        if (isequal (op, 'le'))       z = (x <= y)      ; end

                    end

                    assert (z == result) ;

                    Results (k, 1+ first, 1+ last, 1+ id) = result ;
                end
            end
        end
    end

    % fprintf ('\n=================================================== summary:\n') ;

%{
    for k = 1:length (ops)
        op = ops {k} ;
        fprintf ('\n============================== %s\n', op) ;
        for first = 0:1
            for last = 0:1
                A.values (1) = first ;
                A.values (end) = last ;
                for id = 0:1
                    % no terminal
                    result = Results (k, 1+ first, 1+ last, 1+ id) ; 
                    fprintf ('first: %d  last: %d id: %d op: %6s result: %d\n', first, last, id, op, result) ;
                end
            end
        end
    end
%}

end

fprintf ('\ntest108: all tests passed\n') ;
