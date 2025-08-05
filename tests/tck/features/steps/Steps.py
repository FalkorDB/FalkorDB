import sys
import os
import ast

if 'TCK_PROTOCOL' in os.environ and os.environ['TCK_PROTOCOL'] == 'BOLT':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../bolt_utils/')
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../redis_utils/')

import assertions
import graphs

resultset = None
exception = None
params = None

def before_feature(context):
    global params
    params = None

@given(u'the binary-tree-1 graph')
def step_impl(context):
    graphs.binary_tree_graph1()

@given(u'the binary-tree-2 graph')
def step_impl(context):
    graphs.binary_tree_graph2()

@given(u'any graph')
def step_impl(context):
    graphs.any_graph()

@given(u'an empty graph')
def step_impl(context):
    graphs.any_graph()

@then(u'parameters are:')
@given(u'parameters are:')
def set_params(context):
    global params

    params = "CYPHER "
    heading_row = context.table.headings
    if context.active_outline is not None:
        heading_row = [item if item[0] != "<" else context.active_outline[item[1:-1]] for item in heading_row]
    params += '='.join(heading_row) + ' '

    for row in context.table:
        params += '='.join(row) + ' '

@given(u'having executed:')
@when(u'having executed:')
@then(u'having executed:')
@when(u'executing control query:')
@when(u'executing query:')
def step_impl(context):
    global resultset
    global exception
    global params

    exception = None
    query = context.text

    if params:
        query = params + query

    try:
        resultset = graphs.query(query.replace("\r", ""))
    except Exception as error:
        resultset = None
        exception = error
    params = None

@then(u'the result should be empty')
def step_impl(context):
    assertions.assert_empty_resultset(resultset)

@then(u'the side effects should be:')
def step_impl(context):
    stat = context.table.headings[0]
    value = int(context.table.headings[1])
    assertions.assert_statistics(resultset, stat, value)

    for row in context.table:
        stat = row[0]
        value = int(row[1])
        assertions.assert_statistics(resultset, stat, value)

@then(u'no side effects')
def step_impl(context):
    assertions.assert_no_modifications(resultset)

@then(u'the result should be, in any order:')
def step_impl(context):
    if exception:
        raise exception
    expected_length = len(context.table.rows)
    assertions.assert_resultset_length(resultset, expected_length)
    assertions.assert_resultsets_equals(resultset, context.table)


@then(u'the result should be:')
@then(u'the result should be, in order:')
def step_impl(context):
    if exception:
        raise exception
    expected_length = len(context.table.rows)
    assertions.assert_resultset_length(resultset, expected_length)
    assertions.assert_resultsets_equals_in_order(resultset, context.table)

@then(u'a SyntaxError should be raised at compile time: NoSingleRelationshipType')
def step_impl(context):
    global exception
    assert exception != None
    assert "Exactly one relationship type" in str(exception)

@then(u'a SyntaxError should be raised at compile time: InvalidParameterUse')
def step_impl(context):
    global exception
    assert exception != None
    assert "Encountered unhandled type" in str(exception)

@then(u'a SyntaxError should be raised at compile time: VariableTypeConflict')
def step_impl(context):
    global exception
    assert exception != None
    assert ("The alias" in str(exception)) or ("return of variable-length" in str(exception))

@then(u'a SyntaxError should be raised at compile time: InvalidRelationshipPattern')
def step_impl(context):
    global exception
    assert exception != None
    assert "Invalid input" in str(exception)

@then(u'a SemanticError should be raised at runtime: MergeReadOwnWrites')
def step_impl(context):
    global exception
    assert exception != None
    assert "Cannot merge" in str(exception)

@then(u'a SyntaxError should be raised at compile time: NegativeIntegerArgument')
@then(u'a SyntaxError should be raised at runtime: NegativeIntegerArgument')
def step_impl(context):
    global exception
    assert exception != None
    assert "must be a positive integer" in str(exception)

@then(u'a TypeError should be raised at compile time: InvalidArgumentType')
def step_impl(context):
    global exception
    assert exception != None
    assert "Type mismatch" in str(exception)

@then(u'a TypeError should be raised at runtime: MapElementAccessByNonString')
def step_impl(context):
    global exception
    assert exception != None
    assert "Type mismatch" in str(exception)

@then(u'a TypeError should be raised at runtime: InvalidElementAccess')
def step_impl(context):
    global exception
    assert exception != None
    assert "Type mismatch" in str(exception)

@then(u'a ArgumentError should be raised at runtime: NumberOutOfRange')
def step_impl(context):
    global exception
    assert exception != None
    assert "ArgumentError" in str(exception) or "Invalid input" in str(exception)

@then(u'a SyntaxError should be raised at compile time: InvalidAggregation')
def step_impl(context):
    global exception
    assert exception != None
    assert "Invalid use of aggregating function" in str(exception)

@then(u'a SyntaxError should be raised at compile time: UndefinedVariable')
def step_impl(context):
    global exception
    assert exception != None
    assert "not defined" in str(exception)

@then(u'a SyntaxError should be raised at compile time: VariableAlreadyBound')
def step_impl(context):
    global exception
    assert exception != None
    assert "can't be redeclared" in str(exception)

@then(u'a TypeError should be raised at any time: InvalidArgumentType')
@then(u'a TypeError should be raised at compile time: ListElementAccessByNonInteger')
@then(u'a SyntaxError should be raised at compile time: ListElementAccessByNonInteger')
def step_impl(context):
    global exception
    assert exception != None
    assert "Type mismatch" in str(exception)

@then(u'a TypeError should be raised at any time: *')
def step_impl(context):
    global exception
    assert exception != None

@then(u'a TypeError should be raised at runtime: InvalidArgumentType')
@then(u'a SyntaxError should be raised at runtime: InvalidArgumentType')
@then(u'a ArgumentError should be raised at runtime: InvalidArgumentType')
@then(u'a SyntaxError should be raised at compile time: InvalidArgumentType')
def step_impl(context):
    global exception
    assert exception != None
    assert "Type mismatch" in str(exception)

@then(u'a SyntaxError should be raised at compile time: UnexpectedSyntax')
def step_impl(context):
    global exception
    assert exception != None
    assert "Invalid input" in str(exception)

@then(u'a SyntaxError should be raised at compile time: InvalidUnicodeCharacter')
def step_impl(context):
    global exception
    assert exception != None
    assert "Invalid input" in str(exception)

@then(u'a TypeError should be raised at runtime: InvalidArgumentValue')
def step_impl(context):
    global exception
    assert exception != None
    assert "Type mismatch" in str(exception)

@then(u'a SyntaxError should be raised at compile time: DifferentColumnsInUnion')
def step_impl(context):
    global exception
    assert exception != None
    assert "must have the same column names." in str(exception)

@then(u'a SyntaxError should be raised at compile time: InvalidClauseComposition')
def step_impl(context):
    global exception
    assert exception != None
    assert "Invalid combination" in str(exception)

@then(u'a SyntaxError should be raised at compile time: NestedAggregation')
def step_impl(context):
    global exception
    assert exception != None
    assert "Can't use aggregate functions inside of aggregate functions" in str(exception)

@then(u'a SyntaxError should be raised at compile time: UnknownFunction')
def step_impl(context):
    global exception
    assert exception != None
    assert "Unknown function" in str(exception)

@then(u'a SyntaxError should be raised at compile time: NonConstantExpression')
def step_impl(context):
    global exception
    assert exception != None
    assert "invalid type" in str(exception)

@then(u'a SyntaxError should be raised at compile time: NoExpressionAlias')
def step_impl(context):
    global exception
    assert exception != None
    assert "must be aliased" in str(exception)

@then(u'a SyntaxError should be raised at compile time: InvalidNumberLiteral')
def step_impl(context):
    global exception
    assert exception != None
    assert "Invalid numeric value" in str(exception)

@then(u'a SyntaxError should be raised at compile time: RequiresDirectedRelationship')
def step_impl(context):
    global exception
    assert exception != None
    assert "Only directed relationships" in str(exception)

@then(u'a SyntaxError should be raised at compile time: RelationshipUniquenessViolation')
def step_impl(context):
    global exception
    assert exception != None
    assert "Cannot use the same relationship variable" in str(exception)

@then(u'a SyntaxError should be raised at compile time: NoVariablesInScope')
def step_impl(context):
    global exception
    assert exception != None
    assert "RETURN * is not allowed when there are no variables in scope" in str(exception)

@then(u'a TypeError should be raised at runtime: InvalidPropertyType')
def step_impl(context):
    global exception
    assert exception != None
    assert "Property values can only be of primitive types or arrays of primitive types" in str(exception)

@then(u'a SyntaxError should be raised at compile time: IntegerOverflow')
def step_impl(context):
    global exception
    assert exception != None
    assert "Integer overflow" in str(exception)

@then(u'a SyntaxError should be raised at compile time: FloatingPointOverflow')
def step_impl(context):
    global exception
    assert exception != None
    assert "Float overflow" in str(exception)

@then(u'a SyntaxError should be raised at compile time: InvalidDelete')
def step_imp(context):
    global exception
    assert exception != None
    assert "DELETE can only be called on nodes, paths and relationships" in str(exception)

@then(u'a SyntaxError should be raised at compile time: ColumnNameConflict')
def step_imp(context):
    global exception
    assert exception != None
    assert "Multiple result columns with the same name are not supported." in str(exception)