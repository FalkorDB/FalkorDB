/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/value.h"
#include "src/util/rmalloc.h"
#include "src/datatypes/set.h"
#include "src/datatypes/array.h"
#include "tests/utils/mock_log.h"
#include "src/datatypes/path/path.h"
#include "src/graph/entities/node.h"
#include "src/graph/entities/edge.h"

void setup() {
	Alloc_Reset();
	Logging_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

void test_numerics() {
	SIValue v;
	char const *str = "12345";
	v = SIValue_FromString(str);
	TEST_ASSERT(v.type == T_DOUBLE);
	TEST_ASSERT(v.doubleval == 12345);

	str = "3.14";
	v = SIValue_FromString(str);
	TEST_ASSERT(v.type == T_DOUBLE);

	// almost equals
	TEST_ASSERT(v.doubleval - 3.14 < 0.0001);

	str = "-9876";
	v = SIValue_FromString(str);
	TEST_ASSERT(v.type == T_DOUBLE);
	TEST_ASSERT(v.doubleval == -9876);

	str = "+1.0E1";
	v = SIValue_FromString(str);
	TEST_ASSERT(v.type == T_DOUBLE);
	TEST_ASSERT(v.doubleval == 10);

	SIValue_Free(v);
}

void test_strings() {
	SIValue v;
	char const *str = "Test!";
	v = SIValue_FromString(str);
	TEST_ASSERT(v.type == T_STRING);
	TEST_ASSERT(strcmp(v.stringval, "Test!") == 0);
	SIValue_Free(v);

	// out of double range
	str = "1.0001e10001";
	v = SIValue_FromString(str);
	TEST_ASSERT(v.type == T_STRING);
	TEST_ASSERT(strcmp(v.stringval, "1.0001e10001") == 0);
	SIValue_Free(v);
}

// idempotence and correctness tests for:
// null, bool, long, double, edge, node, array.
void test_null() {
	SIValue siNull = SI_NullVal();
	SIValue siNullOther = SI_NullVal();
	uint64_t origHashCode = SIValue_HashCode(siNull);
	uint64_t otherHashCode = SIValue_HashCode(siNullOther);
	TEST_ASSERT(origHashCode == otherHashCode);
}

void test_hashBool() {
	SIValue siBool = SI_BoolVal(true);
	SIValue siBoolOther = SI_BoolVal(true);
	uint64_t origHashCode = SIValue_HashCode(siBool);
	uint64_t otherHashCode = SIValue_HashCode(siBoolOther);
	TEST_ASSERT(origHashCode == otherHashCode);

	siBool = SI_BoolVal(false);
	origHashCode = SIValue_HashCode(siBool);
	TEST_ASSERT(origHashCode != otherHashCode);

	siBoolOther = SI_BoolVal(false);
	otherHashCode = SIValue_HashCode(siBoolOther);
	TEST_ASSERT(origHashCode == otherHashCode);
}

void test_hashLong() {
	// INT32 and INT64 tests take too long
	for(int64_t i = 0; i < 100; i++) {
		int num = rand();
		SIValue siInt64 = SI_LongVal(num);
		SIValue siInt64Other = SI_LongVal(num);
		uint64_t origHashCode = SIValue_HashCode(siInt64);
		uint64_t otherHashCode = SIValue_HashCode(siInt64Other);
		TEST_ASSERT(origHashCode == otherHashCode);
	}
}

void test_hashDouble() {
	SIValue siDouble = SI_DoubleVal(3.14);
	SIValue siDoubleOther = SI_DoubleVal(3.14);
	uint64_t origHashCode = SIValue_HashCode(siDouble);
	uint64_t otherHashCode = SIValue_HashCode(siDoubleOther);
	TEST_ASSERT(origHashCode == otherHashCode);

	siDouble = SI_DoubleVal(3.14159265);
	origHashCode = SIValue_HashCode(siDouble);
	TEST_ASSERT(origHashCode != otherHashCode);

	siDoubleOther = SI_DoubleVal(3.14159265);
	otherHashCode = SIValue_HashCode(siDoubleOther);
	TEST_ASSERT(origHashCode == otherHashCode);
}

void test_edge() {
	AttributeSet attr;

	Edge e0;
	e0.id = 0;
	e0.attributes = &attr;
	SIValue edge = SI_Edge(&e0);

	Edge e0Other = e0;
	SIValue edgeOther = SI_Edge(&e0Other);

	// Validate same hashCode for the same value, different address.
	uint64_t origHashCode = SIValue_HashCode(edge);
	uint64_t otherHashCode = SIValue_HashCode(edgeOther);
	TEST_ASSERT(origHashCode == otherHashCode);

	// Change entity id.
	Edge e1 = e0;
	e1.id = 1;
	edgeOther = SI_Edge(&e1);
	otherHashCode = SIValue_HashCode(edgeOther);
	TEST_ASSERT(origHashCode != otherHashCode);
}

void test_node() {
	AttributeSet attr;

	Node n0;
	n0.id = 0;
	n0.attributes = &attr;
	SIValue node = SI_Node(&n0);

	Node n0Other = n0;
	SIValue nodeOther = SI_Node(&n0Other);

	// Validate same hashCode for the same value, different address.
	uint64_t origHashCode = SIValue_HashCode(node);
	uint64_t otherHashCode = SIValue_HashCode(nodeOther);
	TEST_ASSERT(origHashCode == otherHashCode);

	// Change entity id.
	Node n1 = n0;
	n1.id = 1;
	nodeOther = SI_Node(&n1);
	otherHashCode = SIValue_HashCode(nodeOther);
	TEST_ASSERT(origHashCode != otherHashCode);
}

void test_array() {
	SIValue arr = SI_EmptyArray();
	SIValue arrOther = SI_EmptyArray();

	// Test for empty arrays.
	uint64_t origHashCode = SIValue_HashCode(arr);
	uint64_t otherHashCode = SIValue_HashCode(arrOther);
	TEST_ASSERT(origHashCode == otherHashCode);

	SIArray_Append(&arr, SI_LongVal(1));
	origHashCode = SIValue_HashCode(arr);
	TEST_ASSERT(origHashCode != otherHashCode);

	SIArray_Append(&arrOther, SI_LongVal(1));
	otherHashCode = SIValue_HashCode(arrOther);
	TEST_ASSERT(origHashCode == otherHashCode);

	SIValue nestedArr = SI_EmptyArray();
	SIArray_Append(&nestedArr, SI_LongVal(2));
	SIArray_Append(&nestedArr, SI_DoubleVal(3.14));

	SIArray_Append(&arr, nestedArr);
	origHashCode = SIValue_HashCode(arr);
	TEST_ASSERT(origHashCode != otherHashCode);

	SIArray_Append(&arrOther, nestedArr);
	otherHashCode = SIValue_HashCode(arrOther);
	TEST_ASSERT(origHashCode == otherHashCode);

	// Test logic for nested type-checking
	bool contains_double = SIArray_ContainsType(arr, T_DOUBLE);
	TEST_ASSERT(contains_double);

	bool contains_string = SIArray_ContainsType(arr, T_STRING);
	TEST_ASSERT(!contains_string);

	// Checking for multiple types should return true if any match is found
	bool contains = SIArray_ContainsType(arr, (SIType)(T_DOUBLE | T_STRING));
	TEST_ASSERT(contains);

	SIValue_Free(nestedArr);
	SIValue_Free(arr);
	SIValue_Free(arrOther);
}

// test for difference in hash code for the same binary representation
// for different types. The value boolean "true" and the integer value "1"
// have the same binary representation. Given that, their types are different,
// so this test makes sure that their hash code is different
void test_hashLongAndBool() {
	SIValue siInt64 = SI_LongVal(1);
	SIValue siBool = SI_BoolVal(true);
	uint64_t longHashCode = SIValue_HashCode(siInt64);
	uint64_t boolHashCode = SIValue_HashCode(siBool);
	TEST_ASSERT(longHashCode != boolHashCode);

	siInt64 = SI_LongVal(0);
	siBool = SI_BoolVal(false);

	longHashCode = SIValue_HashCode(siInt64);
	boolHashCode = SIValue_HashCode(siBool);
	TEST_ASSERT(longHashCode != boolHashCode);
}

// Test for the same hash code for same semantic value.
void test_hashLongAndDouble() {
	SIValue siInt64 = SI_LongVal(1);
	SIValue siDouble = SI_DoubleVal(1.0);
	uint64_t longHashCode = SIValue_HashCode(siInt64);
	uint64_t boolHashCode = SIValue_HashCode(siDouble);
	TEST_ASSERT(longHashCode == boolHashCode);
}

// Test for entities with same id, different types
void test_edgeAndNode() {
	AttributeSet attr;

	Edge e0;
	e0.id = 0;
	e0.attributes = &attr;
	SIValue edge = SI_Edge(&e0);

	Node n0;
	n0.attributes = &attr;
	SIValue node = SI_Node(&n0);

	TEST_ASSERT(SIValue_HashCode(node) != SIValue_HashCode(edge));
}

void test_set() {
	set *set = Set_New();

	// Set should be empty.
	TEST_ASSERT(Set_Size(set) == 0);

	// Populate set.
	Node n;
	Edge e;
	AttributeSet attr;
	n.id = 0;
	n.attributes = &attr;
	e.id = 0;
	e.attributes = &attr;
	SIValue arr = SI_Array(2);

	Set_Add(set, arr);
	Set_Add(set, SI_Node(&n));
	Set_Add(set, SI_Edge(&e));
	Set_Add(set, SI_NullVal());
	Set_Add(set, SI_LongVal(0));
	Set_Add(set, SI_DoubleVal(1));
	Set_Add(set, SI_BoolVal(true));

	TEST_ASSERT(Set_Size(set) == 7);

	// Make sure all items are in set.
	TEST_ASSERT(Set_Contains(set, arr));
	TEST_ASSERT(Set_Contains(set, SI_Node(&n)));
	TEST_ASSERT(Set_Contains(set, SI_Edge(&e)));
	TEST_ASSERT(Set_Contains(set, SI_NullVal()));
	TEST_ASSERT(Set_Contains(set, SI_LongVal(0)));
	TEST_ASSERT(Set_Contains(set, SI_DoubleVal(1)));
	TEST_ASSERT(Set_Contains(set, SI_BoolVal(true)));

	// Test for none existing items.
	TEST_ASSERT(!Set_Contains(set, SI_BoolVal(false)));
	TEST_ASSERT(!Set_Contains(set, SI_LongVal(2)));
	TEST_ASSERT(!Set_Contains(set, SI_DoubleVal(3)));

	// Try to introduce duplicates.
	Set_Add(set, arr);
	Set_Add(set, SI_Node(&n));
	Set_Add(set, SI_Edge(&e));
	Set_Add(set, SI_NullVal());
	Set_Add(set, SI_LongVal(0));
	Set_Add(set, SI_DoubleVal(1));
	Set_Add(set, SI_BoolVal(true));

	// Set item count shouldn't change.
	uint64_t set_size = Set_Size(set);
	TEST_ASSERT(set_size == 7);

	// Remove items from set.
	Set_Remove(set, arr);
	TEST_ASSERT(!Set_Contains(set, arr));
	TEST_ASSERT(Set_Size(set) == --set_size);

	Set_Remove(set, SI_Node(&n));
	TEST_ASSERT(!Set_Contains(set, SI_Node(&n)));
	TEST_ASSERT(Set_Size(set) == --set_size);

	Set_Remove(set, SI_Edge(&e));
	TEST_ASSERT(!Set_Contains(set, SI_Edge(&e)));
	TEST_ASSERT(Set_Size(set) == --set_size);

	Set_Remove(set, SI_NullVal());
	TEST_ASSERT(!Set_Contains(set, SI_NullVal()));
	TEST_ASSERT(Set_Size(set) == --set_size);

	Set_Remove(set, SI_LongVal(0));
	TEST_ASSERT(!Set_Contains(set, SI_LongVal(0)));
	TEST_ASSERT(Set_Size(set) == --set_size);

	Set_Remove(set, SI_DoubleVal(1));
	TEST_ASSERT(!Set_Contains(set, SI_DoubleVal(1)));
	TEST_ASSERT(Set_Size(set) == --set_size);

	Set_Remove(set, SI_BoolVal(true));
	TEST_ASSERT(!Set_Contains(set, SI_BoolVal(true)));
	TEST_ASSERT(Set_Size(set) == --set_size);

	TEST_ASSERT(Set_Size(set) == 0);
	Set_Free(set);
	
	SIValue_Free(arr);
}

void test_path() {
	Path *path = Path_New(3);
	Path *clone;
	Node *n;
	Edge *e;

	// Path should be empty.
	TEST_ASSERT(Path_NodeCount(path) == 0);
	TEST_ASSERT(Path_EdgeCount(path) == 0);
	TEST_ASSERT(Path_Len(path) == 0);

	// Populate path.
	Node ns[3];
	Edge es[2];
	for (uint i = 0; i < 2; i++) {
		ns[i].id = i;
				
		es[i].id = i;
		es[i].src_id = i;
		es[i].dest_id = i + 1;
	}
	ns[2].id = 2;

	for (uint i = 0; i < 2; i++) {
		Path_AppendNode(path, ns[i]);	
		Path_AppendEdge(path, es[i]);
	}
	Path_AppendNode(path, ns[2]);

	TEST_ASSERT(Path_NodeCount(path) == 3);
	TEST_ASSERT(Path_EdgeCount(path) == 2);
	TEST_ASSERT(Path_Len(path) == 2);

	// Make sure all nodes and edges are in path.
	for (uint i = 0; i < 2; i++) {
		n = Path_GetNode(path, i);
		TEST_ASSERT(n->id == i);		
		e = Path_GetEdge(path, i);
		TEST_ASSERT(e->id == i);
	}
	n = Path_GetNode(path, 2);
	TEST_ASSERT(n->id == 2);

	clone = Path_Clone(path);
	// Make sure all nodes and edges are in path.
	for (uint i = 0; i < 2; i++) {
		n = Path_GetNode(clone, i);
		TEST_ASSERT(n->id == i);
		e = Path_GetEdge(clone, i);
		TEST_ASSERT(e->id == i);
	}
	n = Path_GetNode(clone, 2);
	TEST_ASSERT(n->id == 2);

	Node head = Path_Head(path);
	TEST_ASSERT(head.id == 2);

	Path_Reverse(path);

	// Make sure all nodes and edges are reversed in path.
	for (uint i = 0; i < 2; i++) {
		n = Path_GetNode(path, i);
		TEST_ASSERT(n->id == 2 - i);
		e = Path_GetEdge(path, i);
		TEST_ASSERT(e->id == 1 - i);
	}
	n = Path_GetNode(path, 2);
	TEST_ASSERT(n->id == 0);

	Path_Reverse(path);

	// Remove node from path.
	Node pop_n = Path_PopNode(path);
	TEST_ASSERT(pop_n.id == 2);
	TEST_ASSERT(Path_NodeCount(path) == 2);

	// Remove edge from path.
	Edge pop_e = Path_PopEdge(path);
	TEST_ASSERT(pop_e.id == 1);
	TEST_ASSERT(Path_EdgeCount(path) == 1);

	// Make sure removed node not in path
	bool res = Path_ContainsNode(path, &pop_n);
	TEST_ASSERT(!res);

	// Make sure node is in path
	pop_n.id = 0;
	res = Path_ContainsNode(path, &pop_n);
	TEST_ASSERT(res);

	Path_SetNode(path, 1, pop_n);
	Node *n1 = Path_GetNode(path, 1);
	TEST_ASSERT(n1->id == pop_n.id);

	Path_SetEdge(path, 0, pop_e);
	Edge *e1 = Path_GetEdge(path, 0);
	TEST_ASSERT(e1->id == pop_e.id);

	// Make sure clear path clear all nodes and edges
	Path_Clear(path);
	TEST_ASSERT(Path_NodeCount(path) == 0);
	TEST_ASSERT(Path_EdgeCount(path) == 0);

	Path_Free(path);
	Path_Free(clone);
}

TEST_LIST = {
	{"numerics", test_numerics},
	{"strings", test_strings},
	{"null", test_null},
	{"hashBool", test_hashBool},
	{"hashLong", test_hashLong},
	{"hashDouble", test_hashDouble},
	{"edge", test_edge},
	{"node", test_node},
	{"array", test_array},
	{"hashLongAndBool", test_hashLongAndBool},
	{"hashLongAndDouble", test_hashLongAndDouble},
	{"edgeAndNode", test_edgeAndNode},
	{"set", test_set},
	{"path", test_path},
	{NULL, NULL}
};
