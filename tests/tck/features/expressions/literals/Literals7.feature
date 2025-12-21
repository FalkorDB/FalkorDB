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

Feature: Literals7 - List

  Scenario: [1] Return an empty list
    Given any graph
    When executing query:
      """
      RETURN [] AS literal
      """
    Then the result should be, in any order:
      | literal |
      | []      |
    And no side effects

  Scenario: [2] Return a list containing a boolean
    Given any graph
    When executing query:
      """
      RETURN [false] AS literal
      """
    Then the result should be, in any order:
      | literal |
      | [false] |
    And no side effects

  Scenario: [3] Return a list containing a null
    Given any graph
    When executing query:
      """
      RETURN [null] AS literal
      """
    Then the result should be, in any order:
      | literal |
      | [null]  |
    And no side effects

  Scenario: [4] Return a list containing a integer
    Given any graph
    When executing query:
      """
      RETURN [1] AS literal
      """
    Then the result should be, in any order:
      | literal |
      | [1]     |
    And no side effects

  Scenario: [5] Return a list containing a hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN [-0x162CD4F6] AS literal
      """
    Then the result should be, in any order:
      | literal      |
      | [-372036854] |
    And no side effects

  Scenario: [6] Return a list containing a octal integer
    Given any graph
    When executing query:
      """
      RETURN [02613152366] AS literal
      """
    Then the result should be, in any order:
      | literal     |
      | [372036854] |
    And no side effects

  
  Scenario: [7] Return a list containing a float
    Given any graph
    When executing query:
      """
      RETURN [-.1e-5] AS literal
      """
    Then the result should be, in any order:
      | literal     |
      | [-0.000001] |
    And no side effects

  Scenario: [8] Return a list containing a string
    Given any graph
    When executing query:
      """
      RETURN ['abc, as#?lßdj '] AS literal
      """
    Then the result should be, in any order:
      | literal            |
      | ['abc, as#?lßdj '] |
    And no side effects

  Scenario: [9] Return a list containing an empty lists
    Given any graph
    When executing query:
      """
      RETURN [[]] AS literal
      """
    Then the result should be, in any order:
      | literal |
      | [[]]    |
    And no side effects

  Scenario: [10] Return seven-deep nested empty lists
    Given any graph
    When executing query:
      """
      RETURN [[[[[[[]]]]]]] AS literal
      """
    Then the result should be, in any order:
      | literal        |
      | [[[[[[[]]]]]]] |
    And no side effects

  Scenario: [11] Return 20-deep nested empty lists
    Given any graph
    When executing query:
      """
      RETURN [[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]] AS literal
      """
    Then the result should be, in any order:
      | literal                                  |
      | [[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]] |
    And no side effects

  @crash
  Scenario: [12] Return 40-deep nested empty lists
    Given any graph
    When executing query:
      """
      RETURN [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]] AS literal
      """
    Then the result should be, in any order:
      | literal                                                                          |
      | [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]] |
    And no side effects

  Scenario: [13] Return a list containing an empty map
    Given any graph
    When executing query:
      """
      RETURN [{}] AS literal
      """
    Then the result should be, in any order:
      | literal |
      | [{}]    |
    And no side effects

  Scenario: [14] Return a list containing multiple integer
    Given any graph
    When executing query:
      """
      RETURN [1, -2, 077, 0xA4C, 71034856] AS literal
      """
    Then the result should be, in any order:
      | literal              |
      | [1, -2, 63, 2636, 71034856] |
    And no side effects

### Needs more capable tck-api
#  Scenario: [15] Return a list containing multiple strings
#    Given any graph
#    When executing query:
#      """
#      RETURN ['abc, as#?lßdj ','',"'",",","[a","]"] AS literal
#      """
#    Then the result should be, in any order:
#      | literal                                      |
#      | ['abc, as#?lßdj ', '', '\'', ',', '[a', ']'] |
#    And no side effects

  Scenario: [16] Return a list containing multiple mixed values
    Given any graph
    When executing query:
      """
      RETURN [2E-01, ', as#?lßdj ', null, 71034856, false] AS literal
      """
    Then the result should be, in any order:
      | literal                                     |
      | [0.2, ', as#?lßdj ', null, 71034856, false] |
    And no side effects

  Scenario: [17] Return a list containing real and fake nested lists
    Given any graph
    When executing query:
      """
      RETURN [null, [ ' a ', ' ' ], ' [ a ', ' [ ], ] ', ' [ ', [ ' ' ], ' ] ' ] AS literal
      """
    Then the result should be, in any order:
      | literal                                                        |
      | [null, [' a ', ' '], ' [ a ', ' [ ], ] ', ' [ ', [' '], ' ] '] |
    And no side effects

  StyleCheck
  Scenario: [18] Return a complex list containing multiple mixed and nested values
    Given any graph
    When executing query:
      """
      RETURN [ {
                  id: '0001',
                  type: 'donut',
                  name: 'Cake',
                  ppu: 0.55,
                  batters:
                      {
                          batter:
                              [
                                  { id: '1001', type: 'Regular' },
                                  { id: '1002', type: 'Chocolate' },
                                  { id: '1003', type: 'Blueberry' },
                                  { id: '1004', type: 'Devils Food' }
                              ]
                      },
                  topping:
                      [
                          { id: '5001', type: 'None' },
                          { id: '5002', type: 'Glazed' },
                          { id: '5005', type: 'Sugar' },
                          { id: '5007', type: 'Powdered Sugar' },
                          { id: '5006', type: 'Chocolate Sprinkles' },
                          { id: '5003', type: 'Chocolate' },
                          { id: '5004', type: 'Maple' }
                      ]
              },
              {
                  id: '0002',
                  type: 'donut',
                  name: 'Raised',
                  ppu: 0.55,
                  batters:
                      {
                          batter:
                              [
                                  { id: '1001', type: 'Regular' }
                              ]
                      },
                  topping:
                      [
                          { id: '5001', type: 'None' },
                          { id: '5002', type: 'Glazed' },
                          { id: '5005', type: 'Sugar' },
                          { id: '5003', type: 'Chocolate' },
                          { id: '5004', type: 'Maple' }
                      ]
              },
              {
                  id: '0003',
                  type: 'donut',
                  name: 'Old Fashioned',
                  ppu: 0.55,
                  batters:
                      {
                          batter:
                              [
                                  { id: '1001', type: 'Regular' },
                                  { id: '1002', type: 'Chocolate' }
                              ]
                      },
                  topping:
                      [
                          { id: '5001', type: 'None' },
                          { id: '5002', type: 'Glazed' },
                          { id: '5003', type: 'Chocolate' },
                          { id: '5004', type: 'Maple' }
                      ]
              } ] AS literal
      """
    Then the result should be, in any order:
      | literal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
      | [{id: '0001', type: 'donut', name: 'Cake', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}, {id: '1002', type: 'Chocolate'}, {id: '1003', type: 'Blueberry'}, {id: '1004', type: 'Devils Food'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5005', type: 'Sugar'}, {id: '5007', type: 'Powdered Sugar'}, {id: '5006', type: 'Chocolate Sprinkles'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}, {id: '0002', type: 'donut', name: 'Raised', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5005', type: 'Sugar'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}, {id: '0003', type: 'donut', name: 'Old Fashioned', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}, {id: '1002', type: 'Chocolate'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}] |
    And no side effects

  
  Scenario: [19] Fail on a list containing only a comma
    Given any graph
    When executing query:
      """
      RETURN [, ] AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [20] Fail on a nested list with non-matching brackets
    Given any graph
    When executing query:
      """
      RETURN [[[]] AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [21] Fail on a nested list with missing commas
    Given any graph
    When executing query:
      """
      RETURN [[','[]',']] AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax
