/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../errors/errors.h"
#include "../../datatypes/map.h"
#include <math.h>

#define EARTH_RADIUS 6378140.0
#define DegreeToRadians(d) ((d) * M_PI / 180.0)

SIValue AR_TOPOINT(SIValue *argv, int argc, void *private_data) {
	SIValue map = argv[0];
	SIType t = SI_TYPE(map);

	if(t == T_NULL) return SI_NullVal();

	// expecting input to be a map
	// point({latitude: 32.0705767, longitude: 34.8185946})
	ASSERT(t == T_MAP);

	uint key_count = Map_KeyCount(map);
	if(key_count != 2) {
		ErrorCtx_RaiseRuntimeException("A point map should have 2 elements, latitude and longitude");
		return SI_NullVal();
	}

	SIValue latitude;
	SIValue longitude;

	// make sure lat is present in map
	if(!MAP_GET(map, "latitude", latitude)) {
		ErrorCtx_RaiseRuntimeException("Did not find 'latitude' value in point map");
		return SI_NullVal();
	}
	// make sure lon is present in map
	if(!MAP_GET(map, "longitude", longitude)) {
		ErrorCtx_RaiseRuntimeException("Did not find 'longitude' value in point map");
		return SI_NullVal();
	}
	// validate lat, lon types
	if(!(SI_NUMERIC & SI_TYPE(latitude) && SI_NUMERIC & SI_TYPE(longitude))) {
		ErrorCtx_RaiseRuntimeException("'latitude' and 'longitude' values in point map were not both valid numerics");
		return SI_NullVal();
	}

	// validate latitude is in range [-90,90]
	if(SI_GET_NUMERIC(latitude) > 90 || SI_GET_NUMERIC(latitude) < -90) {
		ErrorCtx_RaiseRuntimeException("latitude should be within the -90 to 90 range");
		return SI_NullVal();
	}

	// validate longitude is in range [-180,180]
	if(SI_GET_NUMERIC(longitude) > 180 || SI_GET_NUMERIC(longitude) < -180) {
		ErrorCtx_RaiseRuntimeException("longitude should be within the -180 to 180 range");
		return SI_NullVal();
	}

	return SI_Point(SI_GET_NUMERIC(latitude), SI_GET_NUMERIC(longitude));
}

SIValue AR_DISTANCE(SIValue *argv, int argc, void *private_data) {
	// compute distance between two points
	// a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
	// c = 2 * atan2( √a, √(1−a) )
	// d = R * c
	// where φ represent the latitudes, and λ represent the longitudes

	SIValue p1 = argv[0];
	SIValue p2 = argv[1];

	// check inputs
	if(SI_TYPE(p1) == T_NULL || SI_TYPE(p2) == T_NULL) return SI_NullVal();

	float lat[2] = { DegreeToRadians(p1.point.latitude),
					 DegreeToRadians(p2.point.latitude)
				   };

	float lon[2] = { DegreeToRadians(p1.point.longitude),
					 DegreeToRadians(p2.point.longitude)
				   };

	float dlat = lat[1] - lat[0];
	float dlon = lon[1] - lon[0];

	// a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
	float a = pow(sin(dlat / 2), 2) + cos(lat[0]) * cos(lat[1]) * pow(sin(dlon / 2), 2);

	// c = 2 * atan2( √a, √(1−a) )
	float c = 2 * atan2(sqrt(a), sqrt(1 - a));

	// d = R * c
	float d = EARTH_RADIUS * c;

	return SI_DoubleVal(d);
}

void Register_PointFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_MAP);
	ret_type = T_NULL | T_POINT;
	func_desc = AR_FuncDescNew("point", AR_TOPOINT, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);


	types = array_new(SIType, 2);
	array_append(types, T_NULL | T_POINT);
	array_append(types, T_NULL | T_POINT);
	ret_type = T_NULL | T_DOUBLE;
	func_desc = AR_FuncDescNew("distance", AR_DISTANCE, 2, 2, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);
}
