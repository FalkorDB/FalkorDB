/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../value.h"

// returns latitude of given point
float Point_lat(SIValue point);

// returns longitude of given point
float Point_lon(SIValue point);

// returns a coordinate (latitude or longitude) of a given point
SIValue Point_GetCoordinate(SIValue point, SIValue key);

