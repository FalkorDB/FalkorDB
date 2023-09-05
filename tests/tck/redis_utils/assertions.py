from numbers import Number
from collections import Counter
from RLTest import Env

from redis.commands.graph import Graph
from redis.commands.graph.node import Node
from redis.commands.graph.edge import Edge
from redis.commands.graph.path import Path


def is_numeric(value):
    # check for value's type to be a number or a string
    if not isinstance(value, (Number, str)):
        return False
    try:
        # value is either number or string, try to convert to float
        float(value)
        # conversion succeed
        return True
    except ValueError:
        # value was a string not representing a number
        return False


def removeQuotes(value):
    value = value.replace("'", "")
    value = value.replace('"', "")
    return value


def toNumeric(value):
    value = float(value)
    if value.is_integer():
        value = int(value)
    return value


def nodeToString(value):
    res = '('
    if value.alias:
        res += value.alias
    if value.labels:
        value.labels.sort()
        res += ':' + ":".join(value.labels)
    if value.properties:
        props = ', '.join(key+': '+str(val)
                          for key, val in value.properties.items())
        if value.labels:
            res += " "
        res += '{' + props + '}'
    res += ')'
    value = res
    return value


def edgeToString(value):
    res = "["
    if value.relation:
        res += ":" + value.relation
    if value.properties:
        props = ', '.join(key+': '+str(val)
                          for key, val in value.properties.items())
        if value.relation:
            res += " "
        res += '{' + props + '}'
    res += ']'
    value = res
    return value


def listToString(listToConvert):
    strValue = '['
    strValue += ", ".join(map(lambda value: toString(value), listToConvert))
    strValue += ']'
    return strValue

def pathToString(pathToConvert):
    strValue = "<"
    nodes_count = pathToConvert.nodes_count()
    for i in range(0, nodes_count - 1):
        node = pathToConvert.get_node(i)
        node_str = nodeToString(node)
        edge = pathToConvert.get_relationship(i)
        edge_str = edgeToString(edge)
        strValue += node_str + "-" + edge_str + "->" if edge.src_node == node.id else node_str + "<-" + edge_str + "-"

    strValue += nodeToString(pathToConvert.get_node(nodes_count - 1)) if nodes_count > 0 else ""
    strValue += ">"
    return strValue

def dictToString(dictToConvert):
    size = len(dictToConvert)
    strValue = '{'
    for idx, item in enumerate(dictToConvert.items()):
        strValue += item[0] + ": "
        strValue += toString(item[1])
        if idx < size - 1:
            strValue += ", "
    strValue += '}'
    return strValue

def toString(value):
    if isinstance(value, bool):
        if value is True:
            return "true"
        elif value is False:
            return "false"
    elif is_numeric(value):
        return str(value)
    elif isinstance(value, str):
        # remove qoutes if any
        return removeQuotes(value)
    # value is a node
    elif isinstance(value, Node):
        return nodeToString(value)
    # value is an edge
    elif isinstance(value, Edge):
        return edgeToString(value)
    elif isinstance(value, list):
        return listToString(value)
    elif isinstance(value, Path):
        return pathToString(value)
    elif isinstance(value, dict):
        return dictToString(value)
    elif value == None:
        return "null"


def prepareActualValue(actualValue):
    # if value is a numeric string or a number, transform to numeric value
    if is_numeric(actualValue):
        actualValue = toNumeric(actualValue)
    # value is string
    elif isinstance(actualValue, str):
        # remove qoutes if any
        actualValue = removeQuotes(actualValue)
    # value is a node
    elif isinstance(actualValue, Node):
        actualValue = nodeToString(actualValue)
    # value is an edge
    elif isinstance(actualValue, Edge):
        actualValue = edgeToString(actualValue)
    elif isinstance(actualValue, list):
        actualValue = listToString(actualValue)
    elif isinstance(actualValue, Path):
        actualValue = pathToString(actualValue)
    elif isinstance(actualValue, dict):
        actualValue = dictToString(actualValue)
    else:
        # actual value is null or boolean
        Env.RTestInstance.currEnv.assertTrue(isinstance(actualValue, (type(None), bool)))
    return actualValue


def prepareExpectedValue(expectedValue):
    # the expected value is always string. Do a string preparation
    expectedValue = removeQuotes(expectedValue)
    # in case of boolean value string
    if expectedValue == "true":
        expectedValue = True
    elif expectedValue == "false":
        expectedValue = False
    elif expectedValue == "null":
        expectedValue = None
    # in case of numeric string
    elif is_numeric(expectedValue):
        expectedValue = toNumeric(expectedValue)
    return expectedValue


def prepare_actual_row(row):
    return tuple(prepareActualValue(cell) for cell in row)


def prepare_expected_row(row):
    return tuple(prepareExpectedValue(cell) for cell in row)


def assert_empty_resultset(resultset):
    Env.RTestInstance.currEnv.assertEquals(len(resultset.result_set), 0)


def assert_statistics(resultset, stat, value):
    if stat == "+nodes":
        Env.RTestInstance.currEnv.assertEquals(resultset.nodes_created, value)
    elif stat == "+relationships":
        Env.RTestInstance.currEnv.assertEquals(resultset.relationships_created, value)
    elif stat == "-relationships":
        Env.RTestInstance.currEnv.assertEquals(resultset.relationships_deleted, value)
    elif stat == "+labels":
        Env.RTestInstance.currEnv.assertEquals(resultset.labels_added, value)
    elif stat == "-labels":
        Env.RTestInstance.currEnv.assertEquals(resultset.labels_removed, value)
    elif stat == "+properties":
        Env.RTestInstance.currEnv.assertEquals(resultset.properties_set, value)
    elif stat == "-properties":
        Env.RTestInstance.currEnv.assertEquals(resultset.properties_removed, value)
    elif stat == "-nodes":
        Env.RTestInstance.currEnv.assertEquals(resultset.nodes_deleted, value)
    else:
        print(stat)
        Env.RTestInstance.currEnv.assertTrue(False)


def assert_no_modifications(resultset):
    Env.RTestInstance.currEnv.assertEquals(sum([resultset.nodes_created, resultset.nodes_deleted,
                resultset.properties_set, resultset.relationships_created,
                resultset.relationships_deleted]), 0)


def assert_resultset_length(resultset, length):
    Env.RTestInstance.currEnv.assertEquals(len(resultset.result_set), length)


def assert_resultsets_equals_in_order(actual, expected):
    rowCount = len(expected.rows)
    # check amount of rows
    assert_resultset_length(actual, rowCount)
    for rowIdx in range(rowCount):
        actualRow = prepare_actual_row(actual.result_set[rowIdx])
        expectedRow = prepare_expected_row(expected.rows[rowIdx])
        # compare rows
        Env.RTestInstance.currEnv.assertEquals(actualRow, expectedRow)


def assert_resultsets_equals(actual, expected):
    # Convert each row to a tuple, and maintain a count of how many times that row appears
    actualCtr = Counter(prepare_actual_row(row) for row in actual.result_set)
    expectedCtr = Counter(prepare_expected_row(row) for row in expected)
    # Validate that the constructed Counters are equal
    Env.RTestInstance.currEnv.assertEquals(actualCtr, expectedCtr)