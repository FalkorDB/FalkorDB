from common import *
from datetime import datetime

GRAPH_ID = "temporal_test"

class testTemporalLocalDateTime(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test_date_component_construction(self):
        test_cases = [
            #({'year': 1984},                                   '1984-01-01'),
            #({'year': 1984, 'month': 10},                      '1984-10-01'),
            ({'year': 1984, 'week': 10},                       '1984-03-05'),
            #({'year': 1984, 'month': 10, 'day': 11},           '1984-10-11'),
            #({'year': 1984, 'week': 10, 'dayOfWeek': 3},       '1984-03-07'),
            #({'year': 1984, 'ordinalDay': 202},                '1984-07-20'),
            #({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45}, '1984-08-14'),
            #({'year': 1984, 'quarter': 3 },                    '1984-07-00'),
        ]

        for idx, (map_input, expected) in enumerate(test_cases):
            # Construct the map string for the Cypher query
            map_entries = []

            print(f"map_input: {map_input}")
            query = f"RETURN date($date)"
            result = self.graph.query(query, {'date': map_input})
            actual = str(result.result_set[0][0])
            print(f"actual: {actual}")
            self.env.assertEquals(actual, expected)

    def _test_localdatetime_component_construction(self):
        test_cases = [
            ({'year': 1984, 'month': 10, 'day': 11, 'hour': 12, 'minute': 31, 'second': 14, 'nanosecond': 789, 'millisecond': 123, 'microsecond': 456}, '1984-10-11 12:31:14'), # "1984-10-11 12:31:14"
            ({'year': 1984, 'month': 10, 'day': 11, 'hour': 12, 'minute': 31, 'second': 14, 'nanosecond': 645876123}, '1984-10-11 12:31:14'),
            ({'year': 1984, 'month': 10, 'day': 11, 'hour': 12, 'minute': 31, 'second': 14, 'nanosecond': 3}, '1984-10-11 12:31:14'),
            ({'year': 1984, 'month': 10, 'day': 11, 'hour': 12, 'minute': 31, 'second': 14, 'microsecond': 645876}, '1984-10-11 12:31:14'),
            ({'year': 1984, 'month': 10, 'day': 11, 'hour': 12, 'minute': 31, 'second': 14, 'millisecond': 645}, '1984-10-11 12:31:14'),
            ({'year': 1984, 'month': 10, 'day': 11, 'hour': 12, 'minute': 31, 'second': 14}, '1984-10-11 12:31:14'),
            ({'year': 1984, 'month': 10, 'day': 11, 'hour': 12, 'minute': 31}, '1984-10-11 12:31:00'),
            ({'year': 1984, 'month': 10, 'day': 11, 'hour': 12}, '1984-10-11 12:00:00'),
            ({'year': 1984, 'month': 10, 'day': 11}, '1984-10-11 00:00:00'),
            ({'year': 1984, 'week': 10, 'dayOfWeek': 3, 'hour': 12, 'minute': 31, 'second': 14, 'nanosecond': 645876123}, '1984-03-07 12:31:14'),
            ({'year': 1984, 'week': 10, 'dayOfWeek': 3, 'hour': 12, 'minute': 31, 'second': 14, 'microsecond': 645876}, '1984-03-07 12:31:14'),
            ({'year': 1984, 'week': 10, 'dayOfWeek': 3, 'hour': 12, 'minute': 31, 'second': 14, 'millisecond': 645}, '1984-03-07 12:31:14'),
            ({'year': 1984, 'week': 10, 'dayOfWeek': 3, 'hour': 12, 'minute': 31, 'second': 14}, '1984-03-07 12:31:14'),
            ({'year': 1984, 'week': 10, 'dayOfWeek': 3, 'hour': 12, 'minute': 31}, '1984-03-07 12:31:00'),
            ({'year': 1984, 'week': 10, 'dayOfWeek': 3, 'hour': 12}, '1984-03-07 12:00:00'),
            ({'year': 1984, 'week': 10, 'dayOfWeek': 3}, '1984-03-07 00:00:00'),
            #({'year': 1984, 'ordinalDay': 202, 'hour': 12, 'minute': 31, 'second': 14, 'nanosecond': 645876123}, '1984-07-20 12:31:14'),
            #({'year': 1984, 'ordinalDay': 202, 'hour': 12, 'minute': 31, 'second': 14, 'microsecond': 645876}, '1984-07-20 12:31:14'),
            #({'year': 1984, 'ordinalDay': 202, 'hour': 12, 'minute': 31, 'second': 14, 'millisecond': 645}, '1984-07-20 12:31:14'),
            #({'year': 1984, 'ordinalDay': 202, 'hour': 12, 'minute': 31, 'second': 14}, '1984-07-20 12:31:14'),
            #({'year': 1984, 'ordinalDay': 202, 'hour': 12, 'minute': 31}, '1984-07-20 12:31:00'),
            #({'year': 1984, 'ordinalDay': 202, 'hour': 12}, '1984-07-20 12:00:00'),
            #({'year': 1984, 'ordinalDay': 202}, '1984-07-20 00:00:00'),
            ({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45, 'hour': 12, 'minute': 31, 'second': 14, 'nanosecond': 645876123}, '1984-08-14 12:31:14'),
            ({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45, 'hour': 12, 'minute': 31, 'second': 14, 'microsecond': 645876}, '1984-08-14 12:31:14'),
            ({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45, 'hour': 12, 'minute': 31, 'second': 14, 'millisecond': 645}, '1984-08-14 12:31:14'),
            ({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45, 'hour': 12, 'minute': 31, 'second': 14}, '1984-08-14 12:31:14'),
            ({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45, 'hour': 12, 'minute': 31}, '1984-08-14 12:31:00'),
            ({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45, 'hour': 12}, '1984-08-14 12:00:00'),
            ({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45}, '1984-08-14 00:00:00'),
            ({'year': 1984}, '1984-01-01 00:00:00')
        ]

        for idx, (map_input, expected) in enumerate(test_cases):
            # Construct the map string for the Cypher query
            map_entries = []

            #print(f"map_input: {map_input}")
            query = f"RETURN localdatetime($date)"
            result = self.graph.query(query, {'date': map_input})
            actual = str(result.result_set[0][0])
            #print(f"actual: {actual}")
            self.env.assertEquals(actual, expected)

    def _test_localdatetime_week_construction(self):
        test_cases = [
            ({'year': 1916, 'week': 1}, '1916-01-03 00:00:00'),
            ({'year': 1916, 'week': 52}, '1916-12-25 00:00:00'),
            ({'year': 1917, 'week': 1}, '1917-01-01 00:00:00'),
            ({'year': 1917, 'week': 10}, '1917-03-05 00:00:00'),
            ({'year': 1917, 'week': 30}, '1917-07-23 00:00:00'),
            ({'year': 1917, 'week': 52}, '1917-12-24 00:00:00'),
            ({'year': 1918, 'week': 1}, '1917-12-31 00:00:00'),
            ({'year': 1918, 'week': 52}, '1918-12-23 00:00:00'),
            ({'year': 1918, 'week': 53}, '1918-12-30 00:00:00'),
            #({'year': 1919, 'week': 1}, '1919-01-04 00:00:00'),
            ({'year': 1919, 'week': 52}, '1919-12-22 00:00:00'),
            ({'year': 1917, 'week': 1, 'dayOfWeek': 2}, '1917-01-02 00:00:00'),
        ]

        for idx, (map_input, expected) in enumerate(test_cases):
            # Construct the map string for the Cypher query
            map_entries = []

            query = f"RETURN localdatetime($date)"
            result = self.graph.query(query, {'date': map_input})
            actual = str(result.result_set[0][0])
            self.env.assertEquals(actual, expected)

