BEGIN { last = "" ; lastlen = 0 }

{ len = length($0) ;
    if (!(lastlen == 0 && len == 0)) {
        print $0 ;
    }
    last = $0 ;
    lastlen = len ;
}
