from numbers import Number
from collections import Counter
from RLTest import Env

from neo4j.graph import Node, Relationship, Path


def nodeToString(value):
    res = '('
    if value.labels:
        labels = list(value.labels)
        labels.sort()
        res += ':' + ":".join(labels)
    if value._properties:
        props = ', '.join(key+': ' + prepareActualValue(val)
                          for key, val in value._properties.items())
        if value.labels:
            res += " "
        res += '{' + props + '}'
    res += ')'
    value = res
    return value


def edgeToString(value):
    res = "["
    if value.type:
        res += ":" + value.type
    if value._properties:
        props = ', '.join(key+': ' + prepareActualValue(val)
                          for key, val in value._properties.items())
        if value.type:
            res += " "
        res += '{' + props + '}'
    res += ']'
    value = res
    return value


def listToString(listToConvert):
    strValue = '['
    strValue += ", ".join(map(lambda value: prepareActualValue(value), listToConvert))
    strValue += ']'
    return strValue

def pathToString(pathToConvert: Path):
    strValue = "<"
    nodes_count = len(pathToConvert.nodes)
    for i in range(0, nodes_count - 1):
        node = pathToConvert.nodes[i]
        node_str = nodeToString(node)
        edge = pathToConvert.relationships[i]
        edge_str = edgeToString(edge)
        strValue += node_str + "-" + edge_str + "->" if edge.start_node == node else node_str + "<-" + edge_str + "-"

    strValue += nodeToString(pathToConvert.nodes[nodes_count - 1]) if nodes_count > 0 else ""
    strValue += ">"
    return strValue

def dictToString(dictToConvert):
    size = len(dictToConvert)
    strValue = '{'
    for idx, item in enumerate(dictToConvert.items()):
        strValue += item[0] + ": "
        strValue += prepareActualValue(item[1])
        if idx < size - 1:
            strValue += ", "
    strValue += '}'
    return strValue


def prepareActualValue(actualValue):
    if isinstance(actualValue, bool):
        actualValue = str(actualValue).lower()
    elif isinstance(actualValue, Number):
        actualValue = str(actualValue)
        if actualValue == "-0.0":
            actualValue = "0.0"
        elif "e+" in actualValue:
            actualValue = actualValue.replace("e+", "e")
        elif "e-" in actualValue:
            e_index = actualValue.find('e')
            zeroes = int(actualValue[e_index+2:]) - 1
            if zeroes < 10:
                num_str = ''
                if actualValue[0] == '-':
                    num_str += '-'
                num_str += '0.' + zeroes * '0' + actualValue[:e_index].replace("-", "").replace(".", "")
                actualValue = num_str
    elif isinstance(actualValue, str):
        actualValue = f"'{actualValue}'"
    elif isinstance(actualValue, Node):
        actualValue = nodeToString(actualValue)
    elif isinstance(actualValue, Relationship):
        actualValue = edgeToString(actualValue)
    elif isinstance(actualValue, list):
        actualValue = listToString(actualValue)
    elif isinstance(actualValue, Path):
        actualValue = pathToString(actualValue)
    elif isinstance(actualValue, dict):
        actualValue = dictToString(actualValue)
    elif actualValue is None:
        actualValue = "null"
    else:
        Env.RTestInstance.currEnv.assertTrue(False)
    return actualValue


def prepare_actual_row(row):
    return tuple(prepareActualValue(cell) for cell in row)


def prepare_expected_row(row):
    return tuple(cell for cell in row)


def assert_empty_resultset(resultset):
    Env.RTestInstance.currEnv.assertEquals(len(resultset.result_set), 0)


def assert_statistics(resultset, stat, value):
    if stat == "+nodes":
        Env.RTestInstance.currEnv.assertEquals(resultset.summary.counters.nodes_created, value)
    elif stat == "+relationships":
        Env.RTestInstance.currEnv.assertEquals(resultset.summary.counters.relationships_created, value)
    elif stat == "-relationships":
        Env.RTestInstance.currEnv.assertEquals(resultset.summary.counters.relationships_deleted, value)
    elif stat == "+labels":
        Env.RTestInstance.currEnv.assertEquals(resultset.summary.counters.labels_added, value)
    elif stat == "-labels":
        Env.RTestInstance.currEnv.assertEquals(resultset.summary.counters.labels_removed, value)
    elif stat == "+properties":
        Env.RTestInstance.currEnv.assertEquals(resultset.summary.counters.properties_set, value)
    elif stat == "-properties":
        pass
        # Env.RTestInstance.currEnv.assertEquals(resultset.summary.counters.properties_removed, value)
    elif stat == "-nodes":
        Env.RTestInstance.currEnv.assertEquals(resultset.summary.counters.nodes_deleted, value)
    else:
        print(stat)
        Env.RTestInstance.currEnv.assertTrue(False)


def assert_no_modifications(resultset):
    Env.RTestInstance.currEnv.assertEquals(sum([resultset.summary.counters.nodes_created, resultset.summary.counters.nodes_deleted,
                resultset.summary.counters.properties_set, resultset.summary.counters.relationships_created,
                resultset.summary.counters.relationships_deleted]), 0)


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