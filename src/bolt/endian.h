/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <arpa/inet.h>
#if defined ( __linux__ ) || defined ( __GNU__ )

    // on Linux/GNU, use the GNU functions
    #include <endian.h>
    #include <netinet/in.h>
    #define htonll(x) htobe64(x)
    #define ntohll(x) be64toh(x)

#elif defined ( __MACH__ ) && defined ( __APPLE__ )

    // otherwise, on the Mac, use the MACH functions

#else

    // No functions available

#endif