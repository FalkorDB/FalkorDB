/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../value.h"

// adds two temporal values, where one operand must be a duration
// returns a new temporal value of the same type as the non-Duration operand
// valid combinations:
//   - Duration + Duration/Date/Datetime/Time
//   - Duration/Date/Datetime/Time + Duration
// invalid combinations (error):
//   - Date + Date, etc.
SIValue Temporal_AddDuration
(
	SIValue a,  // Temporal lhs
	SIValue b   // Temporal rhs
);

// subtracts one temporal value from another
// where one operand must be a duration
// returns a new temporal value of the same type as the non-Duration operand
// valid combinations:
//   - Duration/Date/Datetime/Time - Duration
//   - Duration - Duration
// invalid combinations (error):
//   - Duration - Date/Datetime/Time
//   - Date - Date, Datetime - Datetime, etc`
SIValue Temporal_SubDuration
(
	SIValue a,  // Temporal lhs
	SIValue b   // Temporal rhs
);

