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

Feature: Temporal2 - Create Temporal Values from a String

  
  Scenario Outline: [1] Should parse date from string
    Given any graph
    When executing query:
      """
      RETURN date(<str>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | str          | result       |
      | '2015-07-21' | '2015-07-21' |
      | '20150721'   | '2015-07-21' |
      | '2015-07'    | '2015-07-01' |
      | '201507'     | '2015-07-01' |
      | '2015-W30-2' | '2015-07-21' |
      | '2015W302'   | '2015-07-21' |
      | '2015-W30'   | '2015-07-20' |
      | '2015W30'    | '2015-07-20' |
      | '2015-202'   | '2015-07-21' |
      | '2015202'    | '2015-07-21' |
      | '2015'       | '2015-01-01' |

  
  Scenario Outline: [2] Should parse local time from string
    Given any graph
    When executing query:
      """
      RETURN localtime(<str>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | str            | result         |
      | '21:40:32.142' | '21:40:32.142' |
      | '214032.142'   | '21:40:32.142' |
      | '21:40:32'     | '21:40:32'     |
      | '214032'       | '21:40:32'     |
      | '21:40'        | '21:40'        |
      | '2140'         | '21:40'        |
      | '21'           | '21:00'        |

  
  Scenario Outline: [3] Should parse time from string
    Given any graph
    When executing query:
      """
      RETURN time(<str>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | str                 | result               |
      | '21:40:32.142+0100' | '21:40:32.142+01:00' |
      | '214032.142Z'       | '21:40:32.142Z'      |
      | '21:40:32+01:00'    | '21:40:32+01:00'     |
      | '214032-0100'       | '21:40:32-01:00'     |
      | '21:40-01:30'       | '21:40-01:30'        |
      | '2140-00:00'        | '21:40Z'             |
      | '2140-02'           | '21:40-02:00'        |
      | '22+18:00'          | '22:00+18:00'        |

  
  Scenario Outline: [4] Should parse local date time from string
    Given any graph
    When executing query:
      """
      RETURN localdatetime(<str>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | str                       | result                    |
      | '2015-07-21T21:40:32.142' | '2015-07-21T21:40:32.142' |
      | '2015-W30-2T214032.142'   | '2015-07-21T21:40:32.142' |
      | '2015-202T21:40:32'       | '2015-07-21T21:40:32'     |
      | '2015T214032'             | '2015-01-01T21:40:32'     |
      | '20150721T21:40'          | '2015-07-21T21:40'        |
      | '2015-W30T2140'           | '2015-07-20T21:40'        |
      | '2015202T21'              | '2015-07-21T21:00'        |

  
  Scenario Outline: [5] Should parse date time from string
    Given any graph
    When executing query:
      """
      RETURN datetime(<str>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | str                            | result                          |
      | '2015-07-21T21:40:32.142+0100' | '2015-07-21T21:40:32.142+01:00' |
      | '2015-W30-2T214032.142Z'       | '2015-07-21T21:40:32.142Z'      |
      | '2015-202T21:40:32+01:00'      | '2015-07-21T21:40:32+01:00'     |
      | '2015T214032-0100'             | '2015-01-01T21:40:32-01:00'     |
      | '20150721T21:40-01:30'         | '2015-07-21T21:40-01:30'        |
      | '2015-W30T2140-00:00'          | '2015-07-20T21:40Z'             |
      | '2015-W30T2140-02'             | '2015-07-20T21:40-02:00'        |
      | '2015202T21+18:00'             | '2015-07-21T21:00+18:00'        |

  
  Scenario Outline: [6] Should parse date time with named time zone from string
    Given any graph
    When executing query:
      """
      RETURN datetime(<str>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | str                                               | result                                               |
      | '2015-07-21T21:40:32.142+02:00[Europe/Stockholm]' | '2015-07-21T21:40:32.142+02:00[Europe/Stockholm]'    |
      | '2015-07-21T21:40:32.142+0845[Australia/Eucla]'   | '2015-07-21T21:40:32.142+08:45[Australia/Eucla]'     |
      | '2015-07-21T21:40:32.142-04[America/New_York]'    | '2015-07-21T21:40:32.142-04:00[America/New_York]'    |
      | '2015-07-21T21:40:32.142[Europe/London]'          | '2015-07-21T21:40:32.142+01:00[Europe/London]'       |
      | '1818-07-21T21:40:32.142[Europe/Stockholm]'       | '1818-07-21T21:40:32.142+00:53:28[Europe/Stockholm]' |

  
  Scenario Outline: [7] Should parse duration from string
    Given any graph
    When executing query:
      """
      RETURN duration(<str>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | str                        | result                     |
      | 'P14DT16H12M'              | 'P14DT16H12M'              |
      | 'P5M1.5D'                  | 'P5M1DT12H'                |
      | 'P0.75M'                   | 'P22DT19H51M49.5S'         |
      | 'PT0.75M'                  | 'PT45S'                    |
      | 'P2.5W'                    | 'P17DT12H'                 |
      | 'P12Y5M14DT16H12M70S'      | 'P12Y5M14DT16H13M10S'      |
      | 'P2012-02-02T14:37:21.545' | 'P2012Y2M2DT14H37M21.545S' |
