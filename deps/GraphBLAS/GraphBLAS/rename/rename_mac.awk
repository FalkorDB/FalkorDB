{
    gbname = $4
    gsub (/^_GxB_/, "GxB_", gbname) ;
    gsub (/^_GrB_/, "GrB_", gbname) ;
    gsub (/^_GB_/, "GB_", gbname) ;
    gbrename = gbname
    gsub (/GxB/, "GxM", gbrename) ;
    gsub (/GrB/, "GrM", gbrename) ;
    gsub (/GB/, "GM", gbrename) ;
    if (length (gbname) > 0) {
        printf "#define %s %s\n", gbname, gbrename
    }
}
