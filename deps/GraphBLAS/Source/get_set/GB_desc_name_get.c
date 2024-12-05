//------------------------------------------------------------------------------
// GB_desc_name_get: get the name of a descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

#define DNAME(d)                    \
{                                   \
    if (desc == d)                  \
    {                               \
        return (#d) ;               \
    }                               \
}

const char *GB_desc_name_get
(
    GrB_Descriptor desc
)
{ 

    DNAME (GrB_NULL        ) ;
    DNAME (GrB_DESC_T1     ) ;
    DNAME (GrB_DESC_T0     ) ;
    DNAME (GrB_DESC_T0T1   ) ;

    DNAME (GrB_DESC_C      ) ;
    DNAME (GrB_DESC_CT1    ) ;
    DNAME (GrB_DESC_CT0    ) ;
    DNAME (GrB_DESC_CT0T1  ) ;

    DNAME (GrB_DESC_S      ) ;
    DNAME (GrB_DESC_ST1    ) ;
    DNAME (GrB_DESC_ST0    ) ;
    DNAME (GrB_DESC_ST0T1  ) ;

    DNAME (GrB_DESC_SC     ) ;
    DNAME (GrB_DESC_SCT1   ) ;
    DNAME (GrB_DESC_SCT0   ) ;
    DNAME (GrB_DESC_SCT0T1 ) ;

    DNAME (GrB_DESC_R      ) ;
    DNAME (GrB_DESC_RT1    ) ;
    DNAME (GrB_DESC_RT0    ) ;
    DNAME (GrB_DESC_RT0T1  ) ;

    DNAME (GrB_DESC_RC     ) ;
    DNAME (GrB_DESC_RCT1   ) ;
    DNAME (GrB_DESC_RCT0   ) ;
    DNAME (GrB_DESC_RCT0T1 ) ;

    DNAME (GrB_DESC_RS     ) ;
    DNAME (GrB_DESC_RST1   ) ;
    DNAME (GrB_DESC_RST0   ) ;
    DNAME (GrB_DESC_RST0T1 ) ;

    DNAME (GrB_DESC_RSC    ) ;
    DNAME (GrB_DESC_RSCT1  ) ;
    DNAME (GrB_DESC_RSCT0  ) ;
    DNAME (GrB_DESC_RSCT0T1) ;

    // user-defined descriptor
    return (desc->user_name_size > 0 ? desc->user_name : NULL) ;
}

