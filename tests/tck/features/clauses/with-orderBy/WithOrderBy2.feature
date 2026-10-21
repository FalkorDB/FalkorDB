#
# Copyright (c) 2015-2022 "Neo Technology,"
# Network Engine for Objects in Lund AB [http://neotechnology.com]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Attribution Notice under the terms of the Apache License 2.0
#
# This work was created by the collective efforts of the openCypher community.
# Without limiting the terms of Section 6, any Derivative Work that is not
# approved by the public consensus process of the openCypher Implementers Group
# should not be described as “Cypher” (and Cypher® is a registered trademark of
# Neo4j Inc.) or as "openCypher". Extensions by implementers or prototypes or
# proposals for change that have been documented or implemented should only be
# described as "implementation extensions to Cypher" or as "proposed changes to
# Cypher that are not yet approved by the openCypher community".
#

#encoding: utf-8

Feature: WithOrderBy2 - Order by a single expression
# LIMIT is used in the following scenarios to surface the effects or WITH ... ORDER BY ...
# which are otherwise lost after the WITH clause according to Cypher semantics

  Scenario Outline: [1] Sort by a boolean expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {bool: true, bool2: true}),
             (:B {bool: false, bool2: false}),
             (:C {bool: false, bool2: true}),
             (:D {bool: true, bool2: true}),
             (:E {bool: true, bool2: false})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 2
      RETURN a
      """
    Then the result should be, in any order:
      | a                              |
      | (:A {bool: true, bool2: true}) |
      | (:D {bool: true, bool2: true}) |
    And no side effects

    Examples:
      | sort                               |
      | NOT (a.bool AND a.bool2)           |
      | NOT (a.bool AND a.bool2) ASC       |
      | NOT (a.bool AND a.bool2) ASCENDING |

  Scenario Outline: [2] Sort by a boolean expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {bool: true, bool2: true}),
             (:B {bool: false, bool2: false}),
             (:C {bool: false, bool2: true}),
             (:D {bool: true, bool2: true}),
             (:E {bool: true, bool2: false})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                |
      | (:B {bool: false, bool2: false}) |
      | (:C {bool: false, bool2: true})  |
      | (:E {bool: true, bool2: false})  |
    And no side effects

    Examples:
      | sort                                |
      | NOT (a.bool AND a.bool2) DESC       |
      | NOT (a.bool AND a.bool2) DESCENDING |

  Scenario Outline: [3] Sort by an integer expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 9, num2: 5}),
             (:B {num: 5, num2: 4}),
             (:C {num: 30, num2: 3}),
             (:D {num: -11, num2: 2}),
             (:E {num: 7054, num2: 1})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                         |
      | (:E {num: 7054, num2: 1}) |
      | (:C {num: 30, num2: 3})   |
      | (:A {num: 9, num2: 5})    |
    And no side effects

    Examples:
      | sort                                  |
      | (a.num2 + (a.num * 2)) * -1           |
      | (a.num2 + (a.num * 2)) * -1 ASC       |
      | (a.num2 + (a.num * 2)) * -1 ASCENDING |

  Scenario Outline: [4] Sort by an integer expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 9, num2: 5}),
             (:B {num: 5, num2: 4}),
             (:C {num: 30, num2: 3}),
             (:D {num: -11, num2: 2}),
             (:E {num: 7054, num2: 1})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                        |
      | (:D {num: -11, num2: 2}) |
      | (:B {num: 5, num2: 4})   |
      | (:A {num: 9, num2: 5})   |
    And no side effects

    Examples:
      | sort                                   |
      | (a.num2 + (a.num * 2)) * -1 DESC       |
      | (a.num2 + (a.num * 2)) * -1 DESCENDING |

  Scenario Outline: [5] Sort by a float expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 5.025648, num2: 1.96357}),
             (:B {num: 30.94857, num2: 0.00002}),
             (:C {num: 30.94856, num2: 0.00002}),
             (:D {num: -11.2943, num2: -8.5007}),
             (:E {num: 7054.008, num2: 948.841})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                   |
      | (:E {num: 7054.008, num2: 948.841}) |
      | (:B {num: 30.94857, num2: 0.00002}) |
      | (:C {num: 30.94856, num2: 0.00002}) |
    And no side effects

    Examples:
      | sort                                   |
      | (a.num + a.num2 * 2) * -1.01           |
      | (a.num + a.num2 * 2) * -1.01 ASC       |
      | (a.num + a.num2 * 2) * -1.01 ASCENDING |

  Scenario Outline: [6] Sort by a float expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 5.025648, num2: 1.96357}),
             (:B {num: 30.94857, num2: 0.00002}),
             (:C {num: 30.94856, num2: 0.00002}),
             (:D {num: -11.2943, num2: -8.5007}),
             (:E {num: 7054.008, num2: 948.841})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                   |
      | (:D {num: -11.2943, num2: -8.5007}) |
      | (:A {num: 5.025648, num2: 1.96357}) |
      | (:C {num: 30.94856, num2: 0.00002}) |
    And no side effects

    Examples:
      | sort                                    |
      | (a.num + a.num2 * 2) * -1.01 DESC       |
      | (a.num + a.num2 * 2) * -1.01 DESCENDING |

  Scenario Outline: [7] Sort by a string expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {name: 'lorem', title: 'dr.'}),
             (:B {name: 'ipsum', title: 'dr.'}),
             (:C {name: 'dolor', title: 'prof.'}),
             (:D {name: 'sit', title: 'dr.'}),
             (:E {name: 'amet', title: 'prof.'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                  |
      | (:A {name: 'lorem', title: 'dr.'}) |
      | (:B {name: 'ipsum', title: 'dr.'}) |
      | (:D {name: 'sit', title: 'dr.'})   |
    And no side effects

    Examples:
      | sort                             |
      | a.title + ' ' + a.name           |
      | a.title + ' ' + a.name ASC       |
      | a.title + ' ' + a.name ASCENDING |

  Scenario Outline: [8] Sort by a string expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {name: 'lorem', title: 'dr.'}),
             (:B {name: 'ipsum', title: 'dr.'}),
             (:C {name: 'dolor', title: 'prof.'}),
             (:D {name: 'sit', title: 'dr.'}),
             (:E {name: 'amet', title: 'prof.'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                    |
      | (:C {name: 'dolor', title: 'prof.'}) |
      | (:E {name: 'amet', title: 'prof.'})  |
      | (:D {name: 'sit', title: 'dr.'})     |
    And no side effects

    Examples:
      | sort                              |
      | a.title + ' ' + a.name DESC       |
      | a.title + ' ' + a.name DESCENDING |

  Scenario Outline: [9] Sort by a list expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {list: [2, -2], list2: [3, -2]}),
             (:B {list: [1, 2], list2: [2, -2]}),
             (:C {list: [300, 0], list2: [1, -2]}),
             (:D {list: [1, -20], list2: [4, -2]}),
             (:E {list: [2, -2, 100], list2: [5, -2]})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                     |
      | (:C {list: [300, 0], list2: [1, -2]}) |
      | (:B {list: [1, 2], list2: [2, -2]})   |
      | (:A {list: [2, -2], list2: [3, -2]})  |
    And no side effects

    Examples:
      | sort                                                             |
      | [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2           |
      | [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASC       |
      | [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASCENDING |

  Scenario Outline: [10] Sort by a list expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {list: [2, -2], list2: [3, -2]}),
             (:B {list: [1, 2], list2: [2, -2]}),
             (:C {list: [300, 0], list2: [1, -2]}),
             (:D {list: [1, -20], list2: [4, -2]}),
             (:E {list: [2, -2, 100], list2: [5, -2]})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                         |
      | (:E {list: [2, -2, 100], list2: [5, -2]}) |
      | (:D {list: [1, -20], list2: [4, -2]})     |
      | (:A {list: [2, -2], list2: [3, -2]})      |
    And no side effects

    Examples:
      | sort                                                              |
      | [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESC       |
      | [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESCENDING |

  @skip
  Scenario Outline: [11] Sort by a date expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
             (:B {date: date({year: 1980, month: 12, day: 24})}),
             (:C {date: date({year: 1984, month: 10, day: 12})}),
             (:D {date: date({year: 1985, month: 5, day: 6})}),
             (:E {date: date({year: 1980, month: 10, day: 24})}),
             (:F {date: date({year: 1984, month: 10, day: 11})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 2
      RETURN a
      """
    Then the result should be, in any order:
      | a                         |
      | (:A {date: '1910-05-06'}) |
      | (:E {date: '1980-10-24'}) |
    And no side effects

    Examples:
      | sort                                              |
      | a.date + duration({months: 1, days: 2})           |
      | a.date + duration({months: 1, days: 2}) ASC       |
      | a.date + duration({months: 1, days: 2}) ASCENDING |

  @skip
  Scenario Outline: [12] Sort by a date expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
             (:B {date: date({year: 1980, month: 12, day: 24})}),
             (:C {date: date({year: 1984, month: 10, day: 12})}),
             (:D {date: date({year: 1985, month: 5, day: 6})}),
             (:E {date: date({year: 1980, month: 10, day: 24})}),
             (:F {date: date({year: 1984, month: 10, day: 11})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 2
      RETURN a
      """
    Then the result should be, in any order:
      | a                         |
      | (:D {date: '1985-05-06'}) |
      | (:C {date: '1984-10-12'}) |
    And no side effects

    Examples:
      | sort                                               |
      | a.date + duration({months: 1, days: 2}) DESC       |
      | a.date + duration({months: 1, days: 2}) DESCENDING |

  @skip
  Scenario Outline: [13] Sort by a local time expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {time: localtime({hour: 10, minute: 35})}),
             (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
             (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
             (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
             (:E {time: localtime({hour: 12, minute: 31, second: 15})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                 |
      | (:A {time: '10:35'})              |
      | (:D {time: '12:30:14.645876123'}) |
      | (:B {time: '12:31:14.645876123'}) |
    And no side effects

    Examples:
      | sort                                      |
      | a.time + duration({minutes: 6})           |
      | a.time + duration({minutes: 6}) ASC       |
      | a.time + duration({minutes: 6}) ASCENDING |

  @skip
  Scenario Outline: [14] Sort by a local time expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {time: localtime({hour: 10, minute: 35})}),
             (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
             (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
             (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
             (:E {time: localtime({hour: 12, minute: 31, second: 15})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                 |
      | (:E {time: '12:31:15'})           |
      | (:C {time: '12:31:14.645876124'}) |
      | (:B {time: '12:31:14.645876123'}) |
    And no side effects

    Examples:
      | sort                                       |
      | a.time + duration({minutes: 6}) DESC       |
      | a.time + duration({minutes: 6}) DESCENDING |

  @skip
  Scenario Outline: [15] Sort by a time expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
             (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
             (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
             (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
             (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                       |
      | (:D {time: '12:35:15+05:00'})           |
      | (:E {time: '12:30:14.645876123+01:01'}) |
      | (:B {time: '12:31:14.645876123+01:00'}) |
    And no side effects

    Examples:
      | sort                                      |
      | a.time + duration({minutes: 6})           |
      | a.time + duration({minutes: 6}) ASC       |
      | a.time + duration({minutes: 6}) ASCENDING |

  @skip
  Scenario Outline: [16] Sort by a time expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
             (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
             (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
             (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
             (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                       |
      | (:A {time: '10:35-08:00'})              |
      | (:C {time: '12:31:14.645876124+01:00'}) |
      | (:B {time: '12:31:14.645876123+01:00'}) |
    And no side effects

    Examples:
      | sort                                       |
      | a.time + duration({minutes: 6}) DESC       |
      | a.time + duration({minutes: 6}) DESCENDING |

  @skip
  Scenario Outline: [17] Sort by a local date time expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
             (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
             (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
             (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
             (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                                |
      | (:C {datetime: '0001-01-01T01:01:01.000000001'}) |
      | (:E {datetime: '1980-12-11T12:31:14'})           |
      | (:A {datetime: '1984-10-11T12:30:14.000000012'}) |
    And no side effects

    Examples:
      | sort                                                   |
      | a.datetime + duration({days: 4, minutes: 6})           |
      | a.datetime + duration({days: 4, minutes: 6}) ASC       |
      | a.datetime + duration({days: 4, minutes: 6}) ASCENDING |

  @skip
  Scenario Outline: [18] Sort by a local date time expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
             (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
             (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
             (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
             (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                                |
      | (:D {datetime: '9999-09-09T09:59:59.999999999'}) |
      | (:B {datetime: '1984-10-11T12:31:14.645876123'}) |
      | (:A {datetime: '1984-10-11T12:30:14.000000012'}) |
    And no side effects

    Examples:
      | sort                                                    |
      | a.datetime + duration({days: 4, minutes: 6}) DESC       |
      | a.datetime + duration({days: 4, minutes: 6}) DESCENDING |

  @skip
  Scenario Outline: [19] Sort by a date time expression in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
             (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
             (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
             (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
             (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                                      |
      | (:C {datetime: '0001-01-01T01:01:01.000000001-11:59'}) |
      | (:E {datetime: '1980-12-11T12:31:14-11:59'})           |
      | (:B {datetime: '1984-10-11T12:31:14.645876123+00:17'}) |
    And no side effects

    Examples:
      | sort                                                   |
      | a.datetime + duration({days: 4, minutes: 6})           |
      | a.datetime + duration({days: 4, minutes: 6}) ASC       |
      | a.datetime + duration({days: 4, minutes: 6}) ASCENDING |

  @skip
  Scenario Outline: [20] Sort by a date time expression in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
             (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
             (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
             (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
             (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 3
      RETURN a
      """
    Then the result should be, in any order:
      | a                                                      |
      | (:D {datetime: '9999-09-09T09:59:59.999999999+11:59'}) |
      | (:A {datetime: '1984-10-11T12:30:14.000000012+00:15'}) |
      | (:B {datetime: '1984-10-11T12:31:14.645876123+00:17'}) |
    And no side effects

    Examples:
      | sort                                                    |
      | a.datetime + duration({days: 4, minutes: 6}) DESC       |
      | a.datetime + duration({days: 4, minutes: 6}) DESCENDING |

  Scenario Outline: [21] Sort by an expression that is only partially orderable on a non-distinct binding table
    Given an empty graph
    And having executed:
      """
      CREATE ({name: 'A'}),
             ({name: 'A'}),
             ({name: 'B'}),
             ({name: 'C'}),
             ({name: 'C'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.name AS name
        ORDER BY a.name + 'C' <dir>
        LIMIT 2
      RETURN name
      """
    Then the result should be, in any order:
      | name |
      | <x>  |
      | <x>  |
    And no side effects

    Examples:
      | dir  | x   |
      | ASC  | 'A' |
      | DESC | 'C' |

  Scenario Outline: [22] Sort by an expression that is only partially orderable on a non-distinct binding table, but used as a grouping key
    Given an empty graph
    And having executed:
      """
      CREATE ({name: 'A'}),
             ({name: 'A'}),
             ({name: 'B'}),
             ({name: 'C'}),
             ({name: 'C'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.name AS name, count(*) AS cnt
        ORDER BY a.name <dir>
        LIMIT 1
      RETURN name, cnt
      """
    Then the result should be, in any order:
      | name | cnt |
      | <x>  | 2   |
    And no side effects

    Examples:
      | dir  | x   |
      | ASC  | 'A' |
      | DESC | 'C' |

  Scenario Outline: [23] Sort by an expression that is only partially orderable on a non-distinct binding table, but used in parts as a grouping key
    Given an empty graph
    And having executed:
      """
      CREATE ({name: 'A'}),
             ({name: 'A'}),
             ({name: 'B'}),
             ({name: 'C'}),
             ({name: 'C'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.name AS name, count(*) AS cnt
        ORDER BY a.name + 'C' <dir>
        LIMIT 1
      RETURN name, cnt
      """
    Then the result should be, in any order:
      | name | cnt |
      | <x>  | 2   |
    And no side effects

    Examples:
      | dir  | x   |
      | ASC  | 'A' |
      | DESC | 'C' |

  @skip
  Scenario Outline: [24] Sort by an expression that is only partially orderable on a non-distinct binding table, but made distinct
    Given an empty graph
    And having executed:
      """
      CREATE ({name: 'A'}),
             ({name: 'A'}),
             ({name: 'B'}),
             ({name: 'C'}),
             ({name: 'C'})
      """
    When executing query:
      """
      MATCH (a)
      WITH DISTINCT a.name AS name
        ORDER BY a.name <dir>
        LIMIT 1
      RETURN *
      """
    Then the result should be, in any order:
      | name |
      | <x>  |
    And no side effects

    Examples:
      | dir  | x   |
      | ASC  | 'A' |
      | DESC | 'C' |

  @skip
  Scenario Outline: [25] Fail on sorting by an aggregation
    Given any graph
    When executing query:
      """
      MATCH (n)
      WITH n.num1 AS foo
        ORDER BY <sort>
      RETURN foo AS foo
      """
    Then a SyntaxError should be raised at compile time: InvalidAggregation

    Examples:
      | sort                                  |
      | count(1)                              |
      | count(n)                              |
      | count(n.num1)                         |
      | count(1 + n.num1)                     |
      | max(n.num2)                           |
      | max(n.num2) ASC                       |
      | max(n.num2) ASCENDING                 |
      | max(n.num2) DESC                      |
      | max(n.num2) DESCENDING                |
      | max(n.num2), n.name                   |
      | max(n.num2) ASC, n.name               |
      | max(n.num2) ASCENDING, n.name         |
      | max(n.num2) DESC, n.name              |
      | max(n.num2) DESCENDING, n.name        |
      | n.name, max(n.num2)                   |
      | n.name ASC, max(n.num2) ASC           |
      | n.name ASC, max(n.num2) DESC          |
      | n.name DESC, max(n.num2) ASC          |
      | n.name DESC, max(n.num2) DESC         |
      | n.name, max(n.num2), n.name2          |
      | n.name, n.name2, max(n.num2)          |
      | n, max(n.num2)                        |
      | n.num1, max(n.num2)                   |
      | n, max(n.num2), n.num1                |
      | n, count(n.num1), max(n.num2), n.num1 |
