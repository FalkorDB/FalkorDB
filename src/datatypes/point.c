/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "point.h"

float Point_lat(SIValue point) {
	ASSERT(SI_TYPE(point) == T_POINT);

	return point.point.latitude;
}

float Point_lon(SIValue point) {
	ASSERT(SI_TYPE(point) == T_POINT);

	return point.point.longitude;
}

SIValue Point_GetCoordinate(SIValue point, SIValue key) {
	ASSERT(SI_TYPE(point) == T_POINT);
	ASSERT(SI_TYPE(key) == T_STRING);

	if(strcmp(key.stringval, "latitude")==0) {
		return SI_DoubleVal(Point_lat(point));
	} else if(strcmp(key.stringval, "longitude")==0) {
		return SI_DoubleVal(Point_lon(point));
	} else {
		return SI_NullVal();
	}
}

