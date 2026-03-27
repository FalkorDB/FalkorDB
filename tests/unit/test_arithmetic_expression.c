/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/arithmetic/funcs.h"
#include "tests/utils/mock_log.h"
#include "src/arithmetic/algebraic_expression.h"

#include <assert.h>
#include <stdlib.h>

void setup();

#define TEST_INIT setup();

#include "acutest.h"

void setup() {
	// use the malloc family for allocations
	Alloc_Reset () ;
	Logging_Reset();

	AR_RegisterFuncs();  // register arithmetic functions
}

void test_ArithmeticExp_Eq() {
	//--------------------------------------------------------------------------
	// check equality of borrow record expressions
	//--------------------------------------------------------------------------

	AR_ExpNode *a = AR_EXP_NewRecordNode () ;
	AR_ExpNode *b = AR_EXP_NewRecordNode () ;

	TEST_ASSERT (AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	//--------------------------------------------------------------------------
	// check equality of constant operand
	//--------------------------------------------------------------------------

	a = AR_EXP_NewConstOperandNode(SI_LongVal (2)) ;
	b = AR_EXP_NewConstOperandNode(SI_LongVal (2)) ;

	TEST_ASSERT (AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	//--------------------------------------------------------------------------
	// check equality of param operand
	//--------------------------------------------------------------------------

	a = AR_EXP_NewParameterOperandNode ("x") ;
	b = AR_EXP_NewParameterOperandNode ("x") ;

	TEST_ASSERT (AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	a = AR_EXP_NewParameterOperandNode ("x") ;
	b = AR_EXP_NewParameterOperandNode ("y") ;

	TEST_ASSERT (!AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	//--------------------------------------------------------------------------
	// check equality of variable expressions
	//--------------------------------------------------------------------------

	a = AR_EXP_NewVariableOperandNode ("X") ;
	b = AR_EXP_NewVariableOperandNode ("X") ;

	TEST_ASSERT (AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	a = AR_EXP_NewVariableOperandNode ("X") ;
	b = AR_EXP_NewVariableOperandNode ("Y") ;

	TEST_ASSERT (!AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	a = AR_EXP_NewVariableOperandNode ("Y") ;
	b = AR_EXP_NewVariableOperandNode ("X") ;

	TEST_ASSERT (!AR_EXP_Equals (a, b));

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	//--------------------------------------------------------------------------
	// check equality of function expressions
	//--------------------------------------------------------------------------

	// a = 1 + 2
	// b = 1 + 2
	a = AR_EXP_NewOpNode("add", true, 2) ;
	b = AR_EXP_NewOpNode("add", true, 2) ;

	AR_EXP_setChild (a, AR_EXP_NewConstOperandNode (SI_LongVal (1)), 0) ;
	AR_EXP_setChild (a, AR_EXP_NewConstOperandNode (SI_LongVal (2)), 1) ;
	AR_EXP_setChild (b, AR_EXP_NewConstOperandNode (SI_LongVal (1)), 0) ;
	AR_EXP_setChild (b, AR_EXP_NewConstOperandNode (SI_LongVal (2)), 1) ;

	TEST_ASSERT (AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	// a = 1 + 2
	// b = 1 + 3
	a = AR_EXP_NewOpNode("add", true, 2) ;
	b = AR_EXP_NewOpNode("add", true, 2) ;

	AR_EXP_setChild (a, AR_EXP_NewConstOperandNode (SI_LongVal (1)), 0) ;
	AR_EXP_setChild (a, AR_EXP_NewConstOperandNode (SI_LongVal (2)), 1) ;
	AR_EXP_setChild (b, AR_EXP_NewConstOperandNode (SI_LongVal (1)), 0) ;
	AR_EXP_setChild (b, AR_EXP_NewConstOperandNode (SI_LongVal (3)), 1) ;

	TEST_ASSERT (!AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	// a = toUpper('a')
	// b = toUpper('a')
	a = AR_EXP_NewOpNode("toUpper", true, 1) ;
	b = AR_EXP_NewOpNode("toUpper", true, 1) ;

	AR_EXP_setChild (a, AR_EXP_NewConstOperandNode (SI_ConstStringVal ("a")), 0) ;
	AR_EXP_setChild (b, AR_EXP_NewConstOperandNode (SI_ConstStringVal ("a")), 0) ;

	TEST_ASSERT (AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;

	// a = toUpper('a')
	// b = toUpper('b')
	a = AR_EXP_NewOpNode("toUpper", true, 1) ;
	b = AR_EXP_NewOpNode("toUpper", true, 1) ;

	AR_EXP_setChild (a, AR_EXP_NewConstOperandNode (SI_ConstStringVal ("a")), 0) ;
	AR_EXP_setChild (b, AR_EXP_NewConstOperandNode (SI_ConstStringVal ("b")), 0) ;

	TEST_ASSERT (!AR_EXP_Equals (a, b)) ;

	AR_EXP_Free (a) ;
	AR_EXP_Free (b) ;
}

TEST_LIST = {
	{"ArithmeticExp_Eq", test_ArithmeticExp_Eq},
	{NULL, NULL}
};

