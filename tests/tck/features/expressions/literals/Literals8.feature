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

Feature: Literals8 - Maps

  Scenario: [1] Return an empty map
    Given any graph
    When executing query:
      """
      RETURN {} AS literal
      """
    Then the result should be, in any order:
      | literal |
      | {}      |
    And no side effects

  Scenario: [2] Return a map containing one value with alphabetic lower case key
    Given any graph
    When executing query:
      """
      RETURN {abc: 1} AS literal
      """
    Then the result should be, in any order:
      | literal |
      | {abc: 1} |
    And no side effects

  Scenario: [3] Return a map containing one value with alphabetic upper case key
    Given any graph
    When executing query:
      """
      RETURN {ABC: 1} AS literal
      """
    Then the result should be, in any order:
      | literal |
      | {ABC: 1} |
    And no side effects

  Scenario: [4] Return a map containing one value with alphabetic mixed case key
    Given any graph
    When executing query:
      """
      RETURN {aBCdeF: 1} AS literal
      """
    Then the result should be, in any order:
      | literal    |
      | {aBCdeF: 1} |
    And no side effects

  Scenario: [5] Return a map containing one value with alphanumeric mixed case key
    Given any graph
    When executing query:
      """
      RETURN {a1B2c3e67: 1} AS literal
      """
    Then the result should be, in any order:
      | literal       |
      | {a1B2c3e67: 1} |
    And no side effects

  Scenario: [6] Return a map containing a boolean
    Given any graph
    When executing query:
      """
      RETURN {k: false} AS literal
      """
    Then the result should be, in any order:
      | literal   |
      | {k: false} |
    And no side effects

  Scenario: [7] Return a map containing a null
    Given any graph
    When executing query:
      """
      RETURN {k: null} AS literal
      """
    Then the result should be, in any order:
      | literal  |
      | {k: null} |
    And no side effects

  Scenario: [8] Return a map containing a integer
    Given any graph
    When executing query:
      """
      RETURN {k: 1} AS literal
      """
    Then the result should be, in any order:
      | literal |
      | {k: 1}   |
    And no side effects

  Scenario: [9] Return a map containing a hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN {F: -0x162CD4F6} AS literal
      """
    Then the result should be, in any order:
      | literal        |
      | {F: -372036854} |
    And no side effects

  Scenario: [10] Return a map containing a octal integer
    Given any graph
    When executing query:
      """
      RETURN {k: 02613152366} AS literal
      """
    Then the result should be, in any order:
      | literal       |
      | {k: 372036854} |
    And no side effects

  
  Scenario: [11] Return a map containing a float
    Given any graph
    When executing query:
      """
      RETURN {k: -.1e-5} AS literal
      """
    Then the result should be, in any order:
      | literal       |
      | {k: -0.000001} |
    And no side effects

  Scenario: [12] Return a map containing a string
    Given any graph
    When executing query:
      """
      RETURN {k: 'ab: c, as#?lßdj '} AS literal
      """
    Then the result should be, in any order:
      | literal               |
      | {k: 'ab: c, as#?lßdj '} |
    And no side effects

  Scenario: [13] Return a map containing an empty map
    Given any graph
    When executing query:
      """
      RETURN {a: {}} AS literal
      """
    Then the result should be, in any order:
      | literal |
      | {a: {}}  |
    And no side effects

  Scenario: [14] Return seven-deep nested maps
    Given any graph
    When executing query:
      """
      RETURN {a1: {a2: {a3: {a4: {a5: {a6: {}}}}}}} AS literal
      """
    Then the result should be, in any order:
      | literal                          |
      | {a1: {a2: {a3: {a4: {a5: {a6: {}}}}}}} |
    And no side effects

  Scenario: [15] Return 20-deep nested maps
    Given any graph
    When executing query:
      """
      RETURN {a1: {a2: {a3: {a4: {a5: {a6: {a7: {a8: {a9: {a10: {a11: {a12: {a13: {a14: {a15: {a16: {a17: {a18: {a19: {}}}}}}}}}}}}}}}}}}}} AS literal
      """
    Then the result should be, in any order:
      | literal                                                                                                     |
      | {a1: {a2: {a3: {a4: {a5: {a6: {a7: {a8: {a9: {a10: {a11: {a12: {a13: {a14: {a15: {a16: {a17: {a18: {a19: {}}}}}}}}}}}}}}}}}}}} |
    And no side effects

  @crash
  Scenario: [16] Return 40-deep nested maps
    Given any graph
    When executing query:
      """
      RETURN {a1: {a2: {a3: {a4: {a5: {a6: {a7: {a8: {a9: {a10: {a11: {a12: {a13: {a14: {a15: {a16: {a17: {a18: {a19: {a20: {a21: {a22: {a23: {a24: {a25: {a26: {a27: {a28: {a29: {a30: {a31: {a32: {a33: {a34: {a35: {a36: {a37: {a38: {a39: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}} AS literal
      """
    Then the result should be, in any order:
      | literal                                                                                                                                                                                                                             |
      | {a1: {a2: {a3: {a4: {a5: {a6: {a7: {a8: {a9: {a10: {a11: {a12: {a13: {a14: {a15: {a16: {a17: {a18: {a19: {a20: {a21: {a22: {a23: {a24: {a25: {a26: {a27: {a28: {a29: {a30: {a31: {a32: {a33: {a34: {a35: {a36: {a37: {a38: {a39: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}} |
    And no side effects

  Scenario: [17] Return a map containing real and fake nested maps
    Given any graph
    When executing query:
      """
      RETURN { a : ' { b : ' , c : { d : ' ' } , d : ' } ' } AS literal
      """
    Then the result should be, in any order:
      | literal                               |
      | {a: ' { b : ', c: {d: ' '}, d: ' } '} |
    And no side effects

  StyleCheck
  Scenario: [18] Return a complex map containing multiple mixed and nested values
    Given any graph
    When executing query:
      """
      RETURN  { data: [ {
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
              } ] } AS literal
      """
    Then the result should be, in any order:
      | literal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
      | {data: [{id: '0001', type: 'donut', name: 'Cake', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}, {id: '1002', type: 'Chocolate'}, {id: '1003', type: 'Blueberry'}, {id: '1004', type: 'Devils Food'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5005', type: 'Sugar'}, {id: '5007', type: 'Powdered Sugar'}, {id: '5006', type: 'Chocolate Sprinkles'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}, {id: '0002', type: 'donut', name: 'Raised', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5005', type: 'Sugar'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}, {id: '0003', type: 'donut', name: 'Old Fashioned', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}, {id: '1002', type: 'Chocolate'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}]} |
    And no side effects

  
  Scenario: [19] Fail on a map containing key starting with a number
    Given any graph
    When executing query:
      """
      RETURN {1B2c3e67:1} AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [20] Fail on a map containing key with symbol
    Given any graph
    When executing query:
      """
      RETURN {k1#k: 1} AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [21] Fail on a map containing key with dot
    Given any graph
    When executing query:
      """
      RETURN {k1.k: 1} AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  Scenario: [22] Fail on a map containing unquoted string
    Given any graph
    When executing query:
      """
      RETURN {k1: k2} AS literal
      """
    Then a SyntaxError should be raised at compile time: UndefinedVariable

  
  Scenario: [23] Fail on a map containing only a comma
    Given any graph
    When executing query:
      """
      RETURN {, } AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [24] Fail on a map containing a value without key
    Given any graph
    When executing query:
      """
      RETURN {1} AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [25] Fail on a map containing a list without key
    Given any graph
    When executing query:
      """
      RETURN {[]} AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [26] Fail on a map containing a map without key
    Given any graph
    When executing query:
      """
      RETURN {{}} AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [27] Fail on a nested map with non-matching braces
    Given any graph
    When executing query:
      """
      RETURN {k: {k: {}} AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax
