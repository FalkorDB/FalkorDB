/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "filter_tree_utils.h"
#include "RG.h"

bool isInFilter(const FT_FilterNode *filter) {
	return (filter->t == FT_N_EXP &&
			filter->exp.exp->type == AR_EXP_OP &&
			strcasecmp(AR_EXP_GetFuncName(filter->exp.exp), "in") == 0);
}

// extracts both origin and radius from a distance filter
// distance(n.location, origin) < radius
bool extractOriginAndRadius(const FT_FilterNode *filter, SIValue *origin,
		SIValue *radius, char **point) {
	// distance (n.location, origin) < radius

	ASSERT(filter != NULL);

	if(filter->t != FT_N_PRED) return false;

	char        *p             =  NULL;
	SIValue     d              =  SI_NullVal();      // radius
	AR_ExpNode  *lhs           =  filter->pred.lhs;
	AR_ExpNode  *rhs           =  filter->pred.rhs;
	AR_ExpNode  *radius_exp    =  NULL;
	AR_ExpNode  *distance_exp  =  NULL;

	// find distance expression
	if(AR_EXP_IsOperation(lhs) &&
	   strcasecmp(AR_EXP_GetFuncName(lhs), "distance") == 0) {
		radius_exp = rhs;
		distance_exp = lhs;
	}

	// could not find 'distance' function call
	if(distance_exp == NULL) return false;

	// make sure radius is constant
	bool scalar = AR_EXP_ReduceToScalar(radius_exp, true, &d);
	if(!scalar) return false;

	if(!(SI_TYPE(d) & SI_NUMERIC)) {
		SIValue_Free(d);
		return false;
	}

	// find origin
	// distance expression should have 2 arguments
	lhs = distance_exp->op.children[0];
	rhs = distance_exp->op.children[1];

	SIValue  l         =  SI_NullVal();
	SIValue  r         =  SI_NullVal();
	bool     res       =  false;
	bool     l_scalar  =  AR_EXP_ReduceToScalar(lhs, true, &l);
	bool     r_scalar  =  AR_EXP_ReduceToScalar(rhs, true, &r);

	if(l_scalar && !r_scalar) {
		res = AR_EXP_IsAttribute(rhs, &p);
		if(point) *point = p;
		if(origin) *origin = l;
		if(radius) *radius = d;
	} else if(!l_scalar && r_scalar) {
		res = AR_EXP_IsAttribute(lhs, &p);
		if(point) *point = p;
		if(origin) *origin = r;
		if(radius) *radius = d;
	} else {
		res = false;
		SIValue_Free(d);
		if(l_scalar) SIValue_Free(l);
		if(r_scalar) SIValue_Free(r);
	}

	return res;
}

// return true if filter performs distance filtering
// distance(n.location, point({lat:1.1, lon:2.2})) < 40
bool isDistanceFilter(const FT_FilterNode *filter) {
	bool res = extractOriginAndRadius(filter, NULL, NULL, NULL);
	if(res) {
		ASSERT(filter->t == FT_N_PRED);
		AST_Operator op = filter->pred.op;
		// make sure filter structure is: distance(point, origin) <= radius
		res = (op == OP_LT || op == OP_LE);
	}

	return res;
}

