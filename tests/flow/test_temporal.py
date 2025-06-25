from common import *
from datetime import datetime, timedelta

GRAPH_ID = "temporal_test"

class testTemporalLocalTime(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    # test localtime construction from individual components
    def test_localtime_component_construction(self):
        test_cases = [
                ({'hour': 12},                                                                                        '12:00:00'),
                ({'hour': 12, 'minute': 31},                                                                          '12:31:00'),
                ({'hour': 12, 'minute': 31, 'second': 14},                                                            '12:31:14'),
                ({'hour': 12, 'minute': 31, 'second': 14, 'millisecond': 645},                                        '12:31:14'),
                ({'hour': 12, 'minute': 31, 'second': 14, 'microsecond': 645876},                                     '12:31:14'),
                ({'hour': 12, 'minute': 31, 'second': 14, 'nanosecond': 645876123},                                   '12:31:14'),
                ({'hour': 12, 'minute': 31, 'second': 14, 'nanosecond': 789, 'millisecond': 123, 'microsecond': 456}, '12:31:14')
        ]

        for idx, (map_input, expected) in enumerate(test_cases):
            # Construct the map string for the Cypher query
            map_entries = []

            query = f"RETURN localtime($time)"
            result = self.graph.query(query, {'time': map_input})
            actual = str(result.result_set[0][0])
            self.env.assertEquals(actual, expected)

    def test_localtime_from_string(self):
        test_cases = [
                ('21',           '21:00:00'),
                ('2140',         '21:40:00'),
                ('21:40',        '21:40:00'),
                ('214032',       '21:40:32'),
                ('21:40:32',     '21:40:32'),
                ('214032.142',   '21:40:32'),
                ('21:40:32.143', '21:40:32')
        ]

        for idx, (str_input, expected) in enumerate(test_cases):
            query = f"RETURN localtime($str)"
            result = self.graph.query(query, {'str': str_input})
            actual = str(result.result_set[0][0])
            self.env.assertEquals(actual, expected)

    def test_localtime_components(self):
        q = """WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d
               RETURN d.hour, d.minute, d.second"""

        res = self.graph.query(q).result_set
        actual_hour   = res[0][0]
        actual_minute = res[0][1]
        actual_second = res[0][2]

        self.env.assertEquals(actual_hour, 12)
        self.env.assertEquals(actual_minute, 31)
        self.env.assertEquals(actual_second, 14)

    def test_localtime_to_from_string(self):
        q = """WITH localtime({hour: 12, minute: 31, second: 14}) AS d
               RETURN toString(d) AS ts, localtime(toString(d)) = d AS b"""

        res = self.graph.query(q).result_set
        ts = res[0][0]
        b  = res[0][1]

        self.env.assertTrue(b)
        self.env.assertEquals(str(ts), '12:31:14')

    def test_localtime_compare(self):
        q = """WITH localtime({hour: 10, minute: 35}) AS x,
                    localtime({hour: 12, minute: 31, second: 14}) AS d
               RETURN x > d, x < d, x >= d, x <= d, x = d"""

        res = self.graph.query(q).result_set
        gt = res[0][0]
        lt = res[0][1]
        ge = res[0][2]
        le = res[0][3]
        e  = res[0][4]

        self.env.assertFalse(gt)
        self.env.assertTrue(lt)
        self.env.assertFalse(ge)
        self.env.assertTrue(le)
        self.env.assertFalse(e)

class testTemporalDate(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    # test date construction from individual components
    def test_date_component_construction(self):
        test_cases = [
            ({'year': 1984},                                   '1984-01-01'),
            ({'year': 1984, 'month': 10},                      '1984-10-01'),
            ({'year': 1984, 'week': 10},                       '1984-03-05'),
            ({'year': 1984, 'month': 10, 'day': 11},           '1984-10-11'),
            ({'year': 1984, 'week': 10, 'dayOfWeek': 3},       '1984-03-07'),
            ({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45}, '1984-08-14'),
            ({'year': 1984, 'quarter': 3 },                    '1984-07-01'),
            #({'year': 1984, 'ordinalDay': 202},                '1984-07-20'),
        ]

        for idx, (map_input, expected) in enumerate(test_cases):
            # Construct the map string for the Cypher query
            map_entries = []

            query = f"RETURN date($date)"
            result = self.graph.query(query, {'date': map_input})
            actual = str(result.result_set[0][0])
            self.env.assertEquals(actual, expected)

    def test_date_from_string(self):
        test_cases = [
          ('2015',       '2015-01-01'),
          ('201507',     '2015-07-01'),
          ('2015202',    '2015-07-21'),
          #('2015W30',    '2015-07-20'),
          ('2015-07',    '2015-07-01'),
          #('2015-202',   '2015-07-21'),
          #('2015W302',   '2015-07-21'),
          ('2015-W30',   '2015-07-20'),
          ('20150721',   '2015-07-21'),
          ('2015-07-21', '2015-07-21'),
          ('2015-W30-2', '2015-07-21')
        ]

        for idx, (str_input, expected) in enumerate(test_cases):
            query = f"RETURN date($str)"
            result = self.graph.query(query, {'str': str_input})
            actual = str(result.result_set[0][0])
            self.env.assertEquals(actual, expected)

    def test_date_components(self):
        q = """WITH date({year: 1984, month:10, day:21}) AS d
               RETURN d.year, d.quarter, d.month, d.week, d.day, d.dayOfWeek,
                      d.dayOfQuarter, d.ordinalDay"""

        res = self.graph.query(q).result_set

        year         = res[0][0]
        quarter      = res[0][1]
        month        = res[0][2]
        week         = res[0][3]
        day          = res[0][4]
        dayOfWeek    = res[0][5]
        dayOfQuarter = res[0][6]
        ordinalDay   = res[0][7]

        self.env.assertEquals(year, 1984)
        self.env.assertEquals(quarter, 4)
        self.env.assertEquals(month, 10)
        self.env.assertEquals(week, 42)
        self.env.assertEquals(day, 21)
        self.env.assertEquals(dayOfWeek, 0)
        self.env.assertEquals(dayOfQuarter, 23)
        self.env.assertEquals(ordinalDay, 295)

    def test_date_to_from_string(self):
        test_cases = [
            ({'year': 1984},                                   '1984-01-01'),
            ({'year': 1984, 'month': 10},                      '1984-10-01'),
            ({'year': 1984, 'week': 10},                       '1984-03-05'),
            ({'year': 1984, 'month': 10, 'day': 11},           '1984-10-11'),
            ({'year': 1984, 'week': 10, 'dayOfWeek': 3},       '1984-03-07'),
            ({'year': 1984, 'quarter': 3, 'dayOfQuarter': 45}, '1984-08-14'),
            ({'year': 1984, 'quarter': 3 },                    '1984-07-01'),
            #({'year': 1984, 'ordinalDay': 202},                '1984-07-20'),
        ]

        for idx, (map_input, expected) in enumerate(test_cases):
            # Construct the map string for the Cypher query
            map_entries = []

            query = """WITH date($comp) AS d
                       RETURN toString(d) AS ts, date(toString(d)) = d AS b"""

            res = self.graph.query(query, {'comp': map_input}).result_set
            ts  = res[0][0]
            b   = res[0][1]

            self.env.assertTrue(b)
            self.env.assertEquals(ts, expected)

    def test_date_compare(self):
        q = """WITH date({year: 1980, month: 12, day: 24}) AS x,
                    date({year: 1984, month: 10, day: 11}) AS d
               RETURN x > d, x < d, x >= d, x <= d, x = d"""

        res = self.graph.query(q).result_set
        gt = res[0][0]
        lt = res[0][1]
        ge = res[0][2]
        le = res[0][3]
        e  = res[0][4]

        self.env.assertFalse(gt)
        self.env.assertTrue(lt)
        self.env.assertFalse(ge)
        self.env.assertTrue(le)
        self.env.assertFalse(e)

        q = """WITH date({year: 1984, month: 10, day: 11}) AS x,
                    date({year: 1984, month: 10, day: 11}) AS d
               RETURN x > d, x < d, x >= d, x <= d, x = d"""

        res = self.graph.query(q).result_set
        gt = res[0][0]
        lt = res[0][1]
        ge = res[0][2]
        le = res[0][3]
        e  = res[0][4]

        self.env.assertFalse(gt)
        self.env.assertFalse(lt)
        self.env.assertTrue(ge)
        self.env.assertTrue(le)
        self.env.assertTrue(e)

class testTemporalLocalDateTime(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test_localdatetime_component_construction(self):
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

    def test_localdatetime_week_construction(self):
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

    def test_localdatetime_components(self):
        q = """WITH localdatetime({year: 1984, month:10, day:21, hour:10, minute:31, second:46}) AS d
               RETURN d.year, d.quarter, d.month, d.week, d.day, d.dayOfWeek,
                      d.dayOfQuarter, d.ordinalDay, d.hour, d.minute, d.second"""

        res = self.graph.query(q).result_set
        print(f"res: {res[0]}")

        year         = res[0][0]
        quarter      = res[0][1]
        month        = res[0][2]
        week         = res[0][3]
        day          = res[0][4]
        dayOfWeek    = res[0][5]
        dayOfQuarter = res[0][6]
        ordinalDay   = res[0][7]
        hour         = res[0][8]
        minute       = res[0][9]
        second       = res[0][10]

        self.env.assertEquals(year, 1984)
        self.env.assertEquals(quarter, 4)
        self.env.assertEquals(month, 10)
        self.env.assertEquals(week, 42)
        self.env.assertEquals(day, 21)
        self.env.assertEquals(dayOfWeek, 0)
        #self.env.assertEquals(dayOfQuarter, 21)
        self.env.assertEquals(ordinalDay, 295)
        self.env.assertEquals(hour, 10)
        self.env.assertEquals(minute, 31)
        self.env.assertEquals(second, 46)

    def test_localdatetime_from_string(self):
        test_cases = [
                #('2015202T21',  '2015-07-21T21:00'),
                #('2015T214032',  '2015-01-01T21:40:32'),
                #('2015-W30T2140',  '2015-07-20T21:40'),
                #('20150721T21:40',  '2015-07-21T21:40'),
                #('2015-202T21:40:32',  '2015-07-21T21:40:32'),
                #('2015-W30-2T214032.142',  '2015-07-21T21:40:32.142'),
                #('2015-07-21T21:40:32.142',  '2015-07-21T21:40:32.142')
        ]

        for idx, (str_input, expected) in enumerate(test_cases):
            query = "RETURN localdatetime($str)"
            result = self.graph.query(query, {'str': str_input})
            actual = str(result.result_set[0][0])
            self.env.assertEquals(actual, expected)

    def test_localdatetime_to_from_string(self):
        query = """WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d
                   RETURN toString(d) AS ts, localdatetime(toString(d)) = d AS b"""

        #res = self.graph.query(query).result_set
        #ts  = res[0][0]
        #b   = res[0][1]

        #self.env.assertTrue(b)
        #self.env.assertEquals(ts, expected)

    def test_localdatetime_compare(self):
        q = """WITH localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14}) AS x,
                    localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d
               RETURN x > d, x < d, x >= d, x <= d, x = d"""

        res = self.graph.query(q).result_set
        gt = res[0][0]
        lt = res[0][1]
        ge = res[0][2]
        le = res[0][3]
        e  = res[0][4]

        self.env.assertFalse(gt)
        self.env.assertTrue(lt)
        self.env.assertFalse(ge)
        self.env.assertTrue(le)
        self.env.assertFalse(e)

        q = """WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS x,
                    localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d
               RETURN x > d, x < d, x >= d, x <= d, x = d"""

        res = self.graph.query(q).result_set
        gt = res[0][0]
        lt = res[0][1]
        ge = res[0][2]
        le = res[0][3]
        e  = res[0][4]

        self.env.assertFalse(gt)
        self.env.assertFalse(lt)
        self.env.assertTrue(ge)
        self.env.assertTrue(le)
        self.env.assertTrue(e)

class testTemporalDuration(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test_duration_component_construction(self):
        test_cases = [
            ( {'years':   2},                                                                    'P2Y'             ),
            ( {'months':  3},                                                                    'P3M'             ),
            ( {'weeks':   1},                                                                    'P7D'             ),
            ( {'hours':   6},                                                                    'PT6H'            ),
            ( {'minutes': 23},                                                                   'PT23M'           ),
            ( {'seconds': 15},                                                                   'PT15S'           ),
            ( {'years': 2, 'months': 3},                                                         'P2Y3M'           ),
            ( {'years': 2, 'months': 3, 'days':  4},                                             'P2Y3M4D'         ),
            ( {'years': 2, 'months': 3, 'days':  4, 'hours':   5},                               'P2Y3M4DT5H'      ),
            ( {'years': 2, 'months': 3, 'days':  4, 'hours':   5,  'minutes': 22},               'P2Y3M4DT5H22M'   ),
            ( {'years': 2, 'months': 3, 'days':  4, 'hours':   5,  'minutes': 22, 'seconds': 7}, 'P2Y3M4DT5H22M7S' ),
            ( {'years': 2, 'weeks':  1, 'hours': 5, 'minutes': 22, 'seconds': 7},                'P2Y7DT5H22M7S'   ),
        ]

        for idx, (map_input, expected) in enumerate(test_cases):
            # construct the map string for the Cypher query
            map_entries = []

            query = f"RETURN tostring(duration($d))"
            result = self.graph.query(query, {'d': map_input})
            actual = str(result.result_set[0][0])
            self.env.assertEquals(actual, expected)

    def test_duration_components(self):
        q = """WITH duration({years: 2, months:3, weeks:1, days:4, hours:5, minutes:22, seconds:7}) AS d
               RETURN d.years, d.months, d.weeks, d.days, d.hours, d.minutes, d.seconds"""

        res = self.graph.query(q).result_set

        years   = res[0][0]
        months  = res[0][1]
        weeks   = res[0][2]
        days    = res[0][3]
        hours   = res[0][4]
        minutes = res[0][5]
        seconds = res[0][6]

        self.env.assertEquals(years,   2)
        self.env.assertEquals(months,  3)
        self.env.assertEquals(weeks,   0)  # weeks seems to move to days
        self.env.assertEquals(days,    11)
        self.env.assertEquals(hours,   5)
        self.env.assertEquals(minutes, 22)
        self.env.assertEquals(seconds, 7)

    def test_duration_compare(self):
        q = """WITH duration({years: 1, months: 11, days: 11, hours: 12, minutes: 31, seconds: 14}) AS x,
                    duration({years: 1, months: 10, days: 11, hours: 12, minutes: 31, seconds: 14}) AS d
               RETURN x > d, x < d, x >= d, x <= d, x = d"""

        res = self.graph.query(q).result_set
        gt = res[0][0]
        lt = res[0][1]
        ge = res[0][2]
        le = res[0][3]
        e  = res[0][4]

        self.env.assertTrue(gt)
        self.env.assertFalse(lt)
        self.env.assertTrue(ge)
        self.env.assertFalse(le)
        self.env.assertFalse(e)

        q = """WITH localdatetime({year: 1, month: 10, day: 11, hour: 12, minute: 31, second: 14}) AS x,
                    localdatetime({year: 1, month: 10, day: 11, hour: 12, minute: 31, second: 14}) AS d
               RETURN x > d, x < d, x >= d, x <= d, x = d"""

        res = self.graph.query(q).result_set
        gt = res[0][0]
        lt = res[0][1]
        ge = res[0][2]
        le = res[0][3]
        e  = res[0][4]

        self.env.assertFalse(gt)
        self.env.assertFalse(lt)
        self.env.assertTrue(ge)
        self.env.assertTrue(le)
        self.env.assertTrue(e)

    def test_duration_add(self):
        #-----------------------------------------------------------------------
        # duration + duration
        #-----------------------------------------------------------------------

        q = """RETURN duration({years:1, months:1, weeks:1, days:1, hours:1, minutes:32, seconds:10}) +
                      duration({years:2, months:2, weeks:2, days:2, hours:2, minutes:34, seconds:12})"""
        res = self.graph.query(q).result_set[0][0]
        self.env.assertEquals(res, timedelta(years=3, months=3, days=24, hours=4, minutes=6, seconds=24))

        return

        #-----------------------------------------------------------------------
        # duration - duration
        #-----------------------------------------------------------------------

        q = "RETURN duration({}) - duration({})"
        res = self.graph.query(q).result_set[0][0]
        #self.env.assertEquals(res, timedelta(seconds=))

        #-----------------------------------------------------------------------
        # date + duration
        #-----------------------------------------------------------------------

        q = "RETURN date({}) + duration({})"
        res = self.graph.query(q).result_set[0][0]
        #self.env.assertEquals(res, timedelta(seconds=))

        #-----------------------------------------------------------------------
        # date - duration
        #-----------------------------------------------------------------------

        q = "RETURN date({}) - duration({})"
        res = self.graph.query(q).result_set[0][0]
        #self.env.assertEquals(res, timedelta(seconds=))

        #-----------------------------------------------------------------------
        # time + duration
        #-----------------------------------------------------------------------

        q = "RETURN time({}) + duration({})"
        res = self.graph.query(q).result_set[0][0]
        #self.env.assertEquals(res, timedelta(seconds=))

        #-----------------------------------------------------------------------
        # time - duration
        #-----------------------------------------------------------------------

        q = "RETURN time({}) - duration({})"
        res = self.graph.query(q).result_set[0][0]
        #self.env.assertEquals(res, timedelta(seconds=))

        #-----------------------------------------------------------------------
        # datetime + duration
        #-----------------------------------------------------------------------

        q = "RETURN datetime({}) + duration({})"
        res = self.graph.query(q).result_set[0][0]
        #self.env.assertEquals(res, timedelta(seconds=))

        #-----------------------------------------------------------------------
        # datetime - duration
        #-----------------------------------------------------------------------

        q = "RETURN datetime({}) - duration({})"
        res = self.graph.query(q).result_set[0][0]
        #self.env.assertEquals(res, timedelta(seconds=))

