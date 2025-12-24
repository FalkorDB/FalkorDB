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

Feature: Temporal10 - Compute Durations Between two Temporal Values

  
  Scenario Outline: [1] Should split between boundaries correctly
    Given any graph
    When executing query:
      """
      WITH duration.between(<d1>, <d2>) AS dur
      RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond
      """
    Then the result should be, in any order:
      | dur   | dur.days | dur.seconds | dur.nanosecondsOfSecond |
      | <dur> | <days>   | <seconds>   | <nanos>                 |
    And no side effects

    Examples:
      | d1                                                   | d2                                                   | dur                | days | seconds | nanos     |
      | localdatetime('2018-01-01T12:00')                    | localdatetime('2018-01-02T10:00')                    | 'PT22H'            | 0    | 79200   | 0         |
      | localdatetime('2018-01-02T10:00')                    | localdatetime('2018-01-01T12:00')                    | 'PT-22H'           | 0    | -79200  | 0         |
      | localdatetime('2018-01-01T10:00:00.2')               | localdatetime('2018-01-02T10:00:00.1')               | 'PT23H59M59.9S'    | 0    | 86399   | 900000000 |
      | localdatetime('2018-01-02T10:00:00.1')               | localdatetime('2018-01-01T10:00:00.2')               | 'PT-23H-59M-59.9S' | 0    | -86400  | 100000000 |
      | datetime('2017-10-28T23:00+02:00[Europe/Stockholm]') | datetime('2017-10-29T04:00+01:00[Europe/Stockholm]') | 'PT6H'             | 0    | 21600   | 0         |
      | datetime('2017-10-29T04:00+01:00[Europe/Stockholm]') | datetime('2017-10-28T23:00+02:00[Europe/Stockholm]') | 'PT-6H'            | 0    | -21600  | 0         |

  
  Scenario Outline: [2] Should compute duration between two temporals
    Given any graph
    When executing query:
      """
      RETURN duration.between(<lhs>, <rhs>) AS duration
      """
    Then the result should be, in any order:
      | duration |
      | <result> |
    And no side effects

    Examples:
      | lhs                                      | rhs                                      | result                    |
      | date('1984-10-11')                       | date('2015-06-24')                       | 'P30Y8M13D'               |
      | date('1984-10-11')                       | localdatetime('2016-07-21T21:45:22.142') | 'P31Y9M10DT21H45M22.142S' |
      | date('1984-10-11')                       | datetime('2015-07-21T21:40:32.142+0100') | 'P30Y9M10DT21H40M32.142S' |
      | date('1984-10-11')                       | localtime('16:30')                       | 'PT16H30M'                |
      | date('1984-10-11')                       | time('16:30+0100')                       | 'PT16H30M'                |
      | localtime('14:30')                       | date('2015-06-24')                       | 'PT-14H-30M'              |
      | localtime('14:30')                       | localdatetime('2016-07-21T21:45:22.142') | 'PT7H15M22.142S'          |
      | localtime('14:30')                       | datetime('2015-07-21T21:40:32.142+0100') | 'PT7H10M32.142S'          |
      | localtime('14:30')                       | localtime('16:30')                       | 'PT2H'                    |
      | localtime('14:30')                       | time('16:30+0100')                       | 'PT2H'                    |
      | time('14:30')                            | date('2015-06-24')                       | 'PT-14H-30M'              |
      | time('14:30')                            | localdatetime('2016-07-21T21:45:22.142') | 'PT7H15M22.142S'          |
      | time('14:30')                            | datetime('2015-07-21T21:40:32.142+0100') | 'PT6H10M32.142S'          |
      | time('14:30')                            | localtime('16:30')                       | 'PT2H'                    |
      | time('14:30')                            | time('16:30+0100')                       | 'PT1H'                    |
      | localdatetime('2015-07-21T21:40:32.142') | date('2015-06-24')                       | 'P-27DT-21H-40M-32.142S'  |
      | localdatetime('2015-07-21T21:40:32.142') | localdatetime('2016-07-21T21:45:22.142') | 'P1YT4M50S'               |
      | localdatetime('2015-07-21T21:40:32.142') | datetime('2015-07-21T21:40:32.142+0100') | 'PT0S'                    |
      | localdatetime('2015-07-21T21:40:32.142') | localtime('16:30')                       | 'PT-5H-10M-32.142S'       |
      | localdatetime('2015-07-21T21:40:32.142') | time('16:30+0100')                       | 'PT-5H-10M-32.142S'       |
      | datetime('2014-07-21T21:40:36.143+0200') | date('2015-06-24')                       | 'P11M2DT2H19M23.857S'     |
      | datetime('2014-07-21T21:40:36.143+0200') | localdatetime('2016-07-21T21:45:22.142') | 'P2YT4M45.999S'           |
      | datetime('2014-07-21T21:40:36.143+0200') | datetime('2015-07-21T21:40:32.142+0100') | 'P1YT59M55.999S'          |
      | datetime('2014-07-21T21:40:36.143+0200') | localtime('16:30')                       | 'PT-5H-10M-36.143S'       |
      | datetime('2014-07-21T21:40:36.143+0200') | time('16:30+0100')                       | 'PT-4H-10M-36.143S'       |

  
  Scenario Outline: [3] Should compute duration between two temporals in months
    Given any graph
    When executing query:
      """
      RETURN duration.inMonths(<lhs>, <rhs>) AS duration
      """
    Then the result should be, in any order:
      | duration |
      | <result> |
    And no side effects

    Examples:
      | lhs                                      | rhs                                      | result   |
      | date('1984-10-11')                       | date('2015-06-24')                       | 'P30Y8M' |
      | date('1984-10-11')                       | localdatetime('2016-07-21T21:45:22.142') | 'P31Y9M' |
      | date('1984-10-11')                       | datetime('2015-07-21T21:40:32.142+0100') | 'P30Y9M' |
      | date('1984-10-11')                       | localtime('16:30')                       | 'PT0S'   |
      | date('1984-10-11')                       | time('16:30+0100')                       | 'PT0S'   |
      | localtime('14:30')                       | date('2015-06-24')                       | 'PT0S'   |
      | localtime('14:30')                       | localdatetime('2016-07-21T21:45:22.142') | 'PT0S'   |
      | localtime('14:30')                       | datetime('2015-07-21T21:40:32.142+0100') | 'PT0S'   |
      | time('14:30')                            | date('2015-06-24')                       | 'PT0S'   |
      | time('14:30')                            | localdatetime('2016-07-21T21:45:22.142') | 'PT0S'   |
      | time('14:30')                            | datetime('2015-07-21T21:40:32.142+0100') | 'PT0S'   |
      | localdatetime('2015-07-21T21:40:32.142') | date('2015-06-24')                       | 'PT0S'   |
      | localdatetime('2015-07-21T21:40:32.142') | localdatetime('2016-07-21T21:45:22.142') | 'P1Y'    |
      | localdatetime('2015-07-21T21:40:32.142') | datetime('2015-07-21T21:40:32.142+0100') | 'PT0S'   |
      | localdatetime('2015-07-21T21:40:32.142') | localtime('16:30')                       | 'PT0S'   |
      | localdatetime('2015-07-21T21:40:32.142') | time('16:30+0100')                       | 'PT0S'   |
      | datetime('2014-07-21T21:40:36.143+0200') | date('2015-06-24')                       | 'P11M'   |
      | datetime('2014-07-21T21:40:36.143+0200') | localdatetime('2016-07-21T21:45:22.142') | 'P2Y'    |
      | datetime('2014-07-21T21:40:36.143+0200') | datetime('2015-07-21T21:40:32.142+0100') | 'P1Y'    |
      | datetime('2014-07-21T21:40:36.143+0200') | localtime('16:30')                       | 'PT0S'   |
      | datetime('2014-07-21T21:40:36.143+0200') | time('16:30+0100')                       | 'PT0S'   |

  
  Scenario Outline: [4] Should compute duration between two temporals in days
    Given any graph
    When executing query:
      """
      RETURN duration.inDays(<lhs>, <rhs>) AS duration
      """
    Then the result should be, in any order:
      | duration |
      | <result> |
    And no side effects

    Examples:
      | lhs                                      | rhs                                      | result    |
      | date('1984-10-11')                       | date('2015-06-24')                       | 'P11213D' |
      | date('1984-10-11')                       | localdatetime('2016-07-21T21:45:22.142') | 'P11606D' |
      | date('1984-10-11')                       | datetime('2015-07-21T21:40:32.142+0100') | 'P11240D' |
      | date('1984-10-11')                       | localtime('16:30')                       | 'PT0S'    |
      | date('1984-10-11')                       | time('16:30+0100')                       | 'PT0S'    |
      | localtime('14:30')                       | date('2015-06-24')                       | 'PT0S'    |
      | localtime('14:30')                       | localdatetime('2016-07-21T21:45:22.142') | 'PT0S'    |
      | localtime('14:30')                       | datetime('2015-07-21T21:40:32.142+0100') | 'PT0S'    |
      | time('14:30')                            | date('2015-06-24')                       | 'PT0S'    |
      | time('14:30')                            | localdatetime('2016-07-21T21:45:22.142') | 'PT0S'    |
      | time('14:30')                            | datetime('2015-07-21T21:40:32.142+0100') | 'PT0S'    |
      | localdatetime('2015-07-21T21:40:32.142') | date('2015-06-24')                       | 'P-27D'   |
      | localdatetime('2015-07-21T21:40:32.142') | localdatetime('2016-07-21T21:45:22.142') | 'P366D'   |
      | localdatetime('2015-07-21T21:40:32.142') | datetime('2015-07-21T21:40:32.142+0100') | 'PT0S'    |
      | localdatetime('2015-07-21T21:40:32.142') | localtime('16:30')                       | 'PT0S'    |
      | localdatetime('2015-07-21T21:40:32.142') | time('16:30+0100')                       | 'PT0S'    |
      | datetime('2014-07-21T21:40:36.143+0200') | date('2015-06-24')                       | 'P337D'   |
      | datetime('2014-07-21T21:40:36.143+0200') | localdatetime('2016-07-21T21:45:22.142') | 'P731D'   |
      | datetime('2014-07-21T21:40:36.143+0200') | datetime('2015-07-21T21:40:32.142+0100') | 'P365D'   |
      | datetime('2014-07-21T21:40:36.143+0200') | localtime('16:30')                       | 'PT0S'    |
      | datetime('2014-07-21T21:40:36.143+0200') | time('16:30+0100')                       | 'PT0S'    |

  
  Scenario Outline: [5] Should compute duration between two temporals in seconds
    Given any graph
    When executing query:
      """
      RETURN duration.inSeconds(<lhs>, <rhs>) AS duration
      """
    Then the result should be, in any order:
      | duration |
      | <result> |
    And no side effects

    Examples:
      | lhs                                      | rhs                                      | result                |
      | date('1984-10-11')                       | date('2015-06-24')                       | 'PT269112H'           |
      | date('1984-10-11')                       | localdatetime('2016-07-21T21:45:22.142') | 'PT278565H45M22.142S' |
      | date('1984-10-11')                       | datetime('2015-07-21T21:40:32.142+0100') | 'PT269781H40M32.142S' |
      | date('1984-10-11')                       | localtime('16:30')                       | 'PT16H30M'            |
      | date('1984-10-11')                       | time('16:30+0100')                       | 'PT16H30M'            |
      | localtime('14:30')                       | date('2015-06-24')                       | 'PT-14H-30M'          |
      | localtime('14:30')                       | localdatetime('2016-07-21T21:45:22.142') | 'PT7H15M22.142S'      |
      | localtime('14:30')                       | datetime('2015-07-21T21:40:32.142+0100') | 'PT7H10M32.142S'      |
      | localtime('14:30')                       | localtime('16:30')                       | 'PT2H'                |
      | localtime('14:30')                       | time('16:30+0100')                       | 'PT2H'                |
      | time('14:30')                            | date('2015-06-24')                       | 'PT-14H-30M'          |
      | time('14:30')                            | localdatetime('2016-07-21T21:45:22.142') | 'PT7H15M22.142S'      |
      | time('14:30')                            | datetime('2015-07-21T21:40:32.142+0100') | 'PT6H10M32.142S'      |
      | time('14:30')                            | localtime('16:30')                       | 'PT2H'                |
      | time('14:30')                            | time('16:30+0100')                       | 'PT1H'                |
      | localdatetime('2015-07-21T21:40:32.142') | date('2015-06-24')                       | 'PT-669H-40M-32.142S' |
      | localdatetime('2015-07-21T21:40:32.142') | localdatetime('2016-07-21T21:45:22.142') | 'PT8784H4M50S'        |
      | localdatetime('2015-07-21T21:40:32.142') | datetime('2015-07-21T21:40:32.142+0100') | 'PT0S'                |
      | localdatetime('2015-07-21T21:40:32.142') | localtime('16:30')                       | 'PT-5H-10M-32.142S'   |
      | localdatetime('2015-07-21T21:40:32.142') | time('16:30+0100')                       | 'PT-5H-10M-32.142S'   |
      | datetime('2014-07-21T21:40:36.143+0200') | date('2015-06-24')                       | 'PT8090H19M23.857S'   |
      | datetime('2014-07-21T21:40:36.143+0200') | localdatetime('2016-07-21T21:45:22.142') | 'PT17544H4M45.999S'   |
      | datetime('2014-07-21T21:40:36.143+0200') | datetime('2015-07-21T21:40:32.142+0100') | 'PT8760H59M55.999S'   |
      | datetime('2014-07-21T21:40:36.143+0200') | localtime('16:30')                       | 'PT-5H-10M-36.143S'   |
      | datetime('2014-07-21T21:40:36.143+0200') | time('16:30+0100')                       | 'PT-4H-10M-36.143S'   |

  
  Scenario: [6] Should compute duration between if they differ only by a fraction of a second and the first comes after the second.
    Given any graph
    When executing query:
      """
      RETURN duration.inSeconds(localdatetime('2014-07-21T21:40:36.143'), localdatetime('2014-07-21T21:40:36.142')) AS d
      """
    Then the result should be, in any order:
      | d           |
      | 'PT-0.001S' |
    And no side effects

  
  Scenario Outline: [7] Should compute negative duration between in big units
    Given any graph
    When executing query:
      """
      RETURN duration.inMonths(<lhs>, <rhs>) AS duration
      """
    Then the result should be, in any order:
      | duration |
      | <result> |
    And no side effects

    Examples:
      | lhs                                      | rhs                                      | result      |
      | date('2018-03-11')                       | date('2016-06-24')                       | 'P-1Y-8M'   |
      | date('2018-07-21')                       | datetime('2016-07-21T21:40:32.142+0100') | 'P-1Y-11M'  |
      | localdatetime('2018-07-21T21:40:32.142') | date('2016-07-21')                       | 'P-2Y'      |
      | datetime('2018-07-21T21:40:36.143+0200') | localdatetime('2016-07-21T21:40:36.143') | 'P-2Y'      |
      | datetime('2018-07-21T21:40:36.143+0500') | datetime('1984-07-21T22:40:36.143+0200') | 'P-33Y-11M' |

  
  Scenario Outline: [8] Should handle durations at daylight saving time day
    Given any graph
    When executing query:
      """
      RETURN duration.inSeconds(<lhs>, <rhs>) AS duration
      """
    Then the result should be, in any order:
      | duration |
      | <result> |
    And no side effects

    Examples:
      | lhs                                                                               | rhs                                                                               | result  |
      | datetime({year: 2017, month: 10, day: 29, hour: 0, timezone: 'Europe/Stockholm'}) | localdatetime({year: 2017, month: 10, day: 29, hour: 4})                          | 'PT5H'  |
      | datetime({year: 2017, month: 10, day: 29, hour: 0, timezone: 'Europe/Stockholm'}) | localtime({hour: 4})                                                              | 'PT5H'  |
      | localdatetime({year: 2017, month: 10, day: 29, hour: 0 })                         | datetime({year: 2017, month: 10, day: 29, hour: 4, timezone: 'Europe/Stockholm'}) | 'PT5H'  |
      | localtime({hour: 0 })                                                             | datetime({year: 2017, month: 10, day: 29, hour: 4, timezone: 'Europe/Stockholm'}) | 'PT5H'  |
      | date({year: 2017, month: 10, day: 29})                                            | datetime({year: 2017, month: 10, day: 29, hour: 4, timezone: 'Europe/Stockholm'}) | 'PT5H'  |
      | datetime({year: 2017, month: 10, day: 29, hour: 0, timezone: 'Europe/Stockholm'}) | date({year: 2017, month: 10, day: 30})                                            | 'PT25H' |

  
  Scenario: [9] Should handle large durations
    Given any graph
    When executing query:
      """
      RETURN duration.between(date('-999999999-01-01'), date('+999999999-12-31')) AS duration
      """
    Then the result should be, in any order:
      | duration             |
      | 'P1999999998Y11M30D' |
    And no side effects

  
  Scenario: [10] Should handle large durations in seconds
    Given any graph
    When executing query:
      """
      RETURN duration.inSeconds(localdatetime('-999999999-01-01'), localdatetime('+999999999-12-31T23:59:59')) AS duration
      """
    Then the result should be, in any order:
      | duration                  |
      | 'PT17531639991215H59M59S' |
    And no side effects

  
  Scenario Outline: [11] Should handle when seconds and subseconds have different signs
    Given any graph
    When executing query:
      """
      RETURN duration.inSeconds(localtime(<lhs>), localtime(<rhs>)) AS duration
      """
    Then the result should be, in any order:
      | duration |
      | <result> |
    And no side effects

    Examples:
      | lhs          | rhs          | result        |
      | '12:34:54.7' | '12:34:54.3' | 'PT-0.4S'     |
      | '12:34:54.3' | '12:34:54.7' | 'PT0.4S'      |
      | '12:34:54.7' | '12:34:55.3' | 'PT0.6S'      |
      | '12:34:54.7' | '12:44:55.3' | 'PT10M0.6S'   |
      | '12:44:54.7' | '12:34:55.3' | 'PT-9M-59.4S' |
      | '12:34:56'   | '12:34:55.7' | 'PT-0.3S'     |
      | '12:34:56'   | '12:44:55.7' | 'PT9M59.7S'   |
      | '12:44:56'   | '12:34:55.7' | 'PT-10M-0.3S' |
      | '12:34:56.3' | '12:34:54.7' | 'PT-1.6S'     |
      | '12:34:54.7' | '12:34:56.3' | 'PT1.6S'      |

  
  Scenario Outline: [12] Should compute durations with no difference
    Given any graph
    When executing query:
      """
      RETURN duration.inSeconds(<value>, <value>) AS duration
      """
    Then the result should be, in any order:
      | duration |
      | 'PT0S'   |
    And no side effects

    Examples:
      | value           |
      | localtime()     |
      | time()          |
      | date()          |
      | localdatetime() |
      | datetime()      |

  
  Scenario Outline: [13] Should propagate null
    Given any graph
    When executing query:
      """
      RETURN <func>(null, null) AS t
      """
    Then the result should be, in any order:
      | t    |
      | null |
    And no side effects

    Examples:
      | func               |
      | duration.between   |
      | duration.inMonths  |
      | duration.inDays    |
      | duration.inSeconds |
