/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/ast/ast_shared.h"
#include "tests/utils/mock_log.h"
#include "src/util/range/string_range.h"
#include "src/util/range/numeric_range.h"

#include <math.h>

void setup() {
	Alloc_Reset();
	Logging_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

//------------------------------------------------------------------------------
// Numeric range
//------------------------------------------------------------------------------

// TEST_F(RangeTest, NumericRangeNew) {
void test_numericRangeNew() {
	NumericRange *r = NumericRange_New();

	TEST_ASSERT(r->valid);
	TEST_ASSERT(r->max == INFINITY);
	TEST_ASSERT(r->min == -INFINITY);
	TEST_ASSERT(!r->include_max);
	TEST_ASSERT(!r->include_min);

	NumericRange_Free(r);
}

void test_numericRangeValidation() {
	NumericRange *r = NumericRange_New();

	// X > 5 AND X < 5
	r->max = 5;
	r->min = 5;
	r->include_max = false;
	r->include_min = false;
	TEST_ASSERT(!NumericRange_IsValid(r));

	// X >= 5 AND X < 5
	r->max = 5;
	r->min = 5;
	r->include_max = false;
	r->include_min = true;
	TEST_ASSERT(!NumericRange_IsValid(r));

	// X >= 5 AND X <= 5
	r->max = 5;
	r->min = 5;
	r->include_max = true;
	r->include_min = true;
	TEST_ASSERT(NumericRange_IsValid(r));

	// X > 5 AND X <= 5
	r->max = 5;
	r->min = 5;
	r->include_max = true;
	r->include_min = false;
	TEST_ASSERT(!NumericRange_IsValid(r));

	// (5, 10)  X > 5 AND x < 10.
	r->max = 10;
	r->min = 5;
	r->include_max = false;
	r->include_min = false;
	TEST_ASSERT(NumericRange_IsValid(r));

	// (5, 10]  X > 5 AND x <= 10.
	r->max = 10;
	r->min = 5;
	r->include_max = true;
	r->include_min = false;
	TEST_ASSERT(NumericRange_IsValid(r));

	// [5, 10)  X >= 5 AND x < 10.
	r->max = 10;
	r->min = 5;
	r->include_max = false;
	r->include_min = true;
	TEST_ASSERT(NumericRange_IsValid(r));

	// [5, 10]  X >= 5 AND x =< 10.
	r->max = 10;
	r->min = 5;
	r->include_max = true;
	r->include_min = true;
	TEST_ASSERT(NumericRange_IsValid(r));

	NumericRange_Free(r);
}

void test_numericTightenRange() {
	NumericRange *r = NumericRange_New();

	// X < 100
	NumericRange_TightenRange(r, OP_LT, 100);
	TEST_ASSERT(r->max == 100);
	TEST_ASSERT(!r->include_max);

	// X <= 100
	NumericRange_TightenRange(r, OP_LE, 100);
	TEST_ASSERT(r->max == 100);
	TEST_ASSERT(!r->include_max);

	// X >= 50
	NumericRange_TightenRange(r, OP_GE, 50);
	TEST_ASSERT(r->min == 50);
	TEST_ASSERT(r->include_min);

	// X > 50
	NumericRange_TightenRange(r, OP_GT, 50);
	TEST_ASSERT(r->min == 50);
	TEST_ASSERT(!r->include_min);

	// 75 <= X >= 75
	NumericRange_TightenRange(r, OP_EQUAL, 75);
	TEST_ASSERT(r->min == 75);
	TEST_ASSERT(r->include_min);
	TEST_ASSERT(r->max == 75);
	TEST_ASSERT(r->include_max);

	TEST_ASSERT(NumericRange_IsValid(r));
	NumericRange_Free(r);
}

void test_numericContainsValue() {
	NumericRange *r = NumericRange_New();
	// -INF < X < INF
	TEST_ASSERT(NumericRange_ContainsValue(r, 100));

	// X <= 100
	NumericRange_TightenRange(r, OP_LE, 100);
	TEST_ASSERT(!NumericRange_ContainsValue(r, 101));
	TEST_ASSERT(NumericRange_ContainsValue(r, 100));
	TEST_ASSERT(NumericRange_ContainsValue(r, -9999));

	// X > -10
	NumericRange_TightenRange(r, OP_GT, -10);
	TEST_ASSERT(!NumericRange_ContainsValue(r, -10));
	TEST_ASSERT(NumericRange_ContainsValue(r, -9));

	// X >= 0 AND X <= 0
	NumericRange_TightenRange(r, OP_EQUAL, 0);
	TEST_ASSERT(!NumericRange_ContainsValue(r, 1));
	TEST_ASSERT(!NumericRange_ContainsValue(r, -1));
	TEST_ASSERT(NumericRange_ContainsValue(r, 0));

	// X > 0
	NumericRange_TightenRange(r, OP_GT, 0);
	TEST_ASSERT(!NumericRange_ContainsValue(r, 0));

	NumericRange_Free(r);
}

//------------------------------------------------------------------------------
// String range
//------------------------------------------------------------------------------

void test_stringRangeNew() {
	StringRange *r = StringRange_New();

	TEST_ASSERT(r->valid);
	TEST_ASSERT(r->max == NULL);
	TEST_ASSERT(r->min == NULL);
	TEST_ASSERT(!r->include_max);
	TEST_ASSERT(!r->include_min);

	StringRange_Free(r);
}

void test_stringRangeValidation() {
	StringRange *r;

	// X > "a" AND X < "a"
	r = StringRange_New();
	StringRange_TightenRange(r, OP_LT, "a");
	StringRange_TightenRange(r, OP_GT, "a");
	TEST_ASSERT(!StringRange_IsValid(r));
	StringRange_Free(r);

	// X >= "a" AND X < "a"
	r = StringRange_New();
	StringRange_TightenRange(r, OP_LT, "a");
	StringRange_TightenRange(r, OP_GE, "a");
	TEST_ASSERT(!StringRange_IsValid(r));
	StringRange_Free(r);

	// X >= "a" AND X <= "a"
	r = StringRange_New();
	StringRange_TightenRange(r, OP_LE, "a");
	StringRange_TightenRange(r, OP_GE, "a");
	TEST_ASSERT(StringRange_IsValid(r));
	StringRange_Free(r);

	// X > "a" AND X <= "a"
	r = StringRange_New();
	StringRange_TightenRange(r, OP_LE, "a");
	StringRange_TightenRange(r, OP_GT, "a");
	TEST_ASSERT(!StringRange_IsValid(r));
	StringRange_Free(r);

	// ("a", "z")  X > "a" AND x < "z".
	r = StringRange_New();
	StringRange_TightenRange(r, OP_LT, "z");
	StringRange_TightenRange(r, OP_GT, "a");
	TEST_ASSERT(StringRange_IsValid(r));
	StringRange_Free(r);

	// ("a", "z"]  X > "a" AND x <= "z".
	r = StringRange_New();
	StringRange_TightenRange(r, OP_LE, "z");
	StringRange_TightenRange(r, OP_GT, "a");
	TEST_ASSERT(StringRange_IsValid(r));
	StringRange_Free(r);

	// ["a", "z")  X >= "a" AND x < "z".
	r = StringRange_New();
	StringRange_TightenRange(r, OP_LT, "z");
	StringRange_TightenRange(r, OP_GE, "a");
	TEST_ASSERT(StringRange_IsValid(r));
	StringRange_Free(r);

	// ["a", "z"]  X >= "a" AND x =< "z".
	r = StringRange_New();
	StringRange_TightenRange(r, OP_LE, "z");
	StringRange_TightenRange(r, OP_GE, "a");
	TEST_ASSERT(StringRange_IsValid(r));
	StringRange_Free(r);
}

void test_stringTightenRange() {
	StringRange *r = StringRange_New();

	// X < "z"
	StringRange_TightenRange(r, OP_LT, "z");
	TEST_ASSERT(strcmp(r->max, "z") == 0);
	TEST_ASSERT(!r->include_max);

	// X <= "z"
	StringRange_TightenRange(r, OP_LE, "z");
	TEST_ASSERT(strcmp(r->max, "z") == 0);
	TEST_ASSERT(!r->include_max);

	// X >= "a"
	StringRange_TightenRange(r, OP_GE, "a");
	TEST_ASSERT(strcmp(r->min, "a") == 0);
	TEST_ASSERT(r->include_min);

	// X > "a"
	StringRange_TightenRange(r, OP_GT, "a");
	TEST_ASSERT(strcmp(r->min, "a") == 0);
	TEST_ASSERT(!r->include_min);

	// "g" <= X >= "g"
	StringRange_TightenRange(r, OP_EQUAL, "g");
	TEST_ASSERT(strcmp(r->min, "g") == 0);
	TEST_ASSERT(r->include_min);
	TEST_ASSERT(strcmp(r->max, "g") == 0);
	TEST_ASSERT(r->include_max);

	TEST_ASSERT(StringRange_IsValid(r));
	StringRange_Free(r);
}

void test_stringContainsValue() {
	StringRange *r = StringRange_New();
	// -INF < X < INF
	TEST_ASSERT(StringRange_ContainsValue(r, "k"));

	// X <= "y"
	StringRange_TightenRange(r, OP_LE, "y");
	TEST_ASSERT(!StringRange_ContainsValue(r, "z"));
	TEST_ASSERT(StringRange_ContainsValue(r, "y"));
	TEST_ASSERT(StringRange_ContainsValue(r, "a"));

	// X > "a"
	StringRange_TightenRange(r, OP_GT, "a");
	TEST_ASSERT(!StringRange_ContainsValue(r, "a"));
	TEST_ASSERT(StringRange_ContainsValue(r, "b"));

	// X >= "k" AND X <= "k"
	StringRange_TightenRange(r, OP_EQUAL, "k");
	TEST_ASSERT(!StringRange_ContainsValue(r, "l"));
	TEST_ASSERT(!StringRange_ContainsValue(r, "j"));
	TEST_ASSERT(StringRange_ContainsValue(r, "k"));

	// X > "k"
	StringRange_TightenRange(r, OP_GT, "k");
	TEST_ASSERT(!StringRange_ContainsValue(r, "k"));

	StringRange_Free(r);
}

TEST_LIST = {
	{"numericRangeValidation", test_numericRangeValidation},
	{"numericTightenRange", test_numericTightenRange},
	{"numericContainsValue", test_numericContainsValue},
	{"stringRangeNew", test_stringRangeNew},
	{"stringRangeValidation", test_stringRangeValidation},
	{"stringTightenRange", test_stringTightenRange},
	{"stringContainsValue", test_stringContainsValue},
	{NULL, NULL}
};
