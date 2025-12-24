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

Feature: Temporal7 - Compare Temporal Values

  
  Scenario Outline: [1] Should compare dates
    Given any graph
    When executing query:
      """
      WITH date(<map>) AS x, date(<map2>) AS d
      RETURN x > d, x < d, x >= d, x <= d, x = d
      """
    Then the result should be, in any order:
      | x > d | x < d | x >= d | x <= d | x = d |
      | <gt>  | <lt>  | <ge>   | <le>   | <e>   |
    And no side effects

    Examples:
      | map                              | map2                             | gt    | lt    | ge    | le   | e     |
      | {year: 1980, month: 12, day: 24} | {year: 1984, month: 10, day: 11} | false | true  | false | true | false |
      | {year: 1984, month: 10, day: 11} | {year: 1984, month: 10, day: 11} | false | false | true  | true | true  |

  
  Scenario Outline: [2] Should compare local times
    Given any graph
    When executing query:
      """
      WITH localtime(<map>) AS x, localtime(<map2>) AS d
      RETURN x > d, x < d, x >= d, x <= d, x = d
      """
    Then the result should be, in any order:
      | x > d | x < d | x >= d | x <= d | x = d |
      | <gt>  | <lt>  | <ge>   | <le>   | <e>   |
    And no side effects

    Examples:
      | map                                                       | map2                                                      | gt    | lt    | ge    | le   | e     |
      | {hour: 10, minute: 35}                                    | {hour: 12, minute: 31, second: 14, nanosecond: 645876123} | false | true  | false | true | false |
      | {hour: 12, minute: 31, second: 14, nanosecond: 645876123} | {hour: 12, minute: 31, second: 14, nanosecond: 645876123} | false | false | true  | true | true  |

  
  Scenario Outline: [3] Should compare times
    Given any graph
    When executing query:
      """
      WITH time(<map>) AS x, time(<map2>) AS d
      RETURN x > d, x < d, x >= d, x <= d, x = d
      """
    Then the result should be, in any order:
      | x > d | x < d | x >= d | x <= d | x = d |
      | <gt>  | <lt>  | <ge>   | <le>   | <e>   |
    And no side effects

    Examples:
      | map                                                                          | map2                                                                         | gt    | lt    | ge    | le   | e     |
      | {hour: 10, minute: 0, timezone: '+01:00'}                                    | {hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'} | false | true  | false | true | false |
      | {hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'} | {hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'} | false | false | true  | true | true  |

  
  Scenario Outline: [4] Should compare local date times
    Given any graph
    When executing query:
      """
      WITH localdatetime(<map>) AS x, localdatetime(<map2>) AS d
      RETURN x > d, x < d, x >= d, x <= d, x = d
      """
    Then the result should be, in any order:
      | x > d | x < d | x >= d | x <= d | x = d |
      | <gt>  | <lt>  | <ge>   | <le>   | <e>   |
    And no side effects

    Examples:
      | map                                                                                       | map2                                                                                      | gt    | lt    | ge    | le   | e     |
      | {year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14}                        | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123} | false | true  | false | true | false |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123} | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123} | false | false | true  | true | true  |

  
  Scenario Outline: [5] Should compare date times
    Given any graph
    When executing query:
      """
      WITH datetime(<map>) AS x, datetime(<map2>) AS d
      RETURN x > d, x < d, x >= d, x <= d, x = d
      """
    Then the result should be, in any order:
      | x > d | x < d | x >= d | x <= d | x = d |
      | <gt>  | <lt>  | <ge>   | <le>   | <e>   |
    And no side effects

    Examples:
      | map                                                                                    | map2                                                                                   | gt    | lt    | ge    | le   | e     |
      | {year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '+00:00'} | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'} | false | true  | false | true | false |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'} | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'} | false | false | true  | true | true  |

  
  Scenario Outline: [6] Should compare durations for equality
    Given any graph
    When executing query:
      """
      WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, <other> AS d
      RETURN x = d
      """
    Then the result should be, in any order:
      | x = d |
      | <e>   |
    And no side effects

    Examples:
      | other                                                                                                    | e     |
      | date({year: 1984, month: 10, day: 11})                                                                   | false |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                     | false |
      | time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'})                       | false |
      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) | false |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'})         | false |
      | duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70})                          | true  |
      | duration({years: 12, months: 5, days: 14, hours: 16, minutes: 13, seconds: 10})                          | true  |
      | duration({years: 12, months: 5, days: 13, hours: 40, minutes: 13, seconds: 10})                          | false |