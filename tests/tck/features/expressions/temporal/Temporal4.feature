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

Feature: Temporal4 - Store Temporal Values
  # Storage tests, cannot be merged into fewer tests due to compatibility reasons

  
  Scenario Outline: [1] Should store date
    Given an empty graph
    When executing query:
      """
      CREATE ({created: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.created
      """
    Then the result should be, in any order:
      | n.created |
      | <result>  |

    Examples:
      | temporal                               | result       |
      | date({year: 1984, month: 10, day: 11}) | '1984-10-11' |

  
  Scenario Outline: [2] Should store date array
    Given an empty graph
    When executing query:
      """
      CREATE ({dates: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.dates
      """
    Then the result should be, in any order:
      | n.dates  |
      | <result> |

    Examples:
      | temporal                                                                                                                 | result                                     |
      | [date({year: 1984, month: 10, day: 12})]                                                                                 | ['1984-10-12']                             |
      | [date({year: 1984, month: 10, day: 13}), date({year: 1984, month: 10, day: 14}), date({year: 1984, month: 10, day: 15})] | ['1984-10-13', '1984-10-14', '1984-10-15'] |

  
  Scenario Outline: [3] Should store local time
    Given an empty graph
    When executing query:
      """
      CREATE ({created: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.created
      """
    Then the result should be, in any order:
      | n.created |
      | <result>  |

    Examples:
      | temporal              | result  |
      | localtime({hour: 12}) | '12:00' |

  
  Scenario Outline: [4] Should store local time array
    Given an empty graph
    When executing query:
      """
      CREATE ({dates: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.dates
      """
    Then the result should be, in any order:
      | n.dates  |
      | <result> |

    Examples:
      | temporal                                                              | result                      |
      | [localtime({hour: 13})]                                               | ['13:00']                   |
      | [localtime({hour: 14}), localtime({hour: 15}), localtime({hour: 16})] | ['14:00', '15:00', '16:00'] |

  
  Scenario Outline: [5] Should store time
    Given an empty graph
    When executing query:
      """
      CREATE ({created: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.created
      """
    Then the result should be, in any order:
      | n.created |
      | <result>  |

    Examples:
      | temporal         | result   |
      | time({hour: 12}) | '12:00Z' |

  
  Scenario Outline: [6] Should store time array
    Given an empty graph
    When executing query:
      """
      CREATE ({dates: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.dates
      """
    Then the result should be, in any order:
      | n.dates  |
      | <result> |

    Examples:
      | temporal                                               | result                         |
      | [time({hour: 13})]                                     | ['13:00Z']                     |
      | [time({hour: 14}), time({hour: 15}), time({hour: 16})] | ['14:00Z', '15:00Z', '16:00Z'] |

  
  Scenario Outline: [7] Should store local date time
    Given an empty graph
    When executing query:
      """
      CREATE ({created: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.created
      """
    Then the result should be, in any order:
      | n.created |
      | <result>  |

    Examples:
      | temporal                    | result             |
      | localdatetime({year: 1912}) | '1912-01-01T00:00' |

  
  Scenario Outline: [8] Should store local date time array
    Given an empty graph
    When executing query:
      """
      CREATE ({dates: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.dates
      """
    Then the result should be, in any order:
      | n.dates  |
      | <result> |

    Examples:
      | temporal                                                                                | result                                                       |
      | [localdatetime({year: 1913})]                                                           | ['1913-01-01T00:00']                                         |
      | [localdatetime({year: 1914}), localdatetime({year: 1915}), localdatetime({year: 1916})] | ['1914-01-01T00:00', '1915-01-01T00:00', '1916-01-01T00:00'] |

  
  Scenario Outline: [9] Should store date time
    Given an empty graph
    When executing query:
      """
      CREATE ({created: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.created
      """
    Then the result should be, in any order:
      | n.created |
      | <result>  |

    Examples:
      | temporal               | result              |
      | datetime({year: 1912}) | '1912-01-01T00:00Z' |

  
  Scenario Outline: [10] Should store date time array
    Given an empty graph
    When executing query:
      """
      CREATE ({dates: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.dates
      """
    Then the result should be, in any order:
      | n.dates  |
      | <result> |

    Examples:
      | temporal                                                                 | result                                                          |
      | [datetime({year: 1913})]                                                 | ['1913-01-01T00:00Z']                                           |
      | [datetime({year: 1914}), datetime({year: 1915}), datetime({year: 1916})] | ['1914-01-01T00:00Z', '1915-01-01T00:00Z', '1916-01-01T00:00Z'] |

  
  Scenario Outline: [11] Should store duration
    Given an empty graph
    When executing query:
      """
      CREATE ({created: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.created
      """
    Then the result should be, in any order:
      | n.created |
      | <result>  |

    Examples:
      | temporal                | result  |
      | duration({seconds: 12}) | 'PT12S' |

  
  Scenario Outline: [12] Should store duration array
    Given an empty graph
    When executing query:
      """
      CREATE ({dates: <temporal>})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
    When executing control query:
      """
      MATCH (n)
      RETURN n.dates
      """
    Then the result should be, in any order:
      | n.dates  |
      | <result> |

    Examples:
      | temporal                                                                    | result                      |
      | [duration({seconds: 13})]                                                   | ['PT13S']                   |
      | [duration({seconds: 14}), duration({seconds: 15}), duration({seconds: 16})] | ['PT14S', 'PT15S', 'PT16S'] |

  
  Scenario Outline: [13] Should propagate null
    Given any graph
    When executing query:
      """
      RETURN <func>(null) AS t
      """
    Then the result should be, in any order:
      | t    |
      | null |
    And no side effects

    Examples:
      | func                      |
      | date                      |
      | date.transaction          |
      | date.statement            |
      | date.realtime             |
      | localtime                 |
      | localtime.transaction     |
      | localtime.statement       |
      | localtime.realtime        |
      | time                      |
      | time.transaction          |
      | time.statement            |
      | time.realtime             |
      | localdatetime             |
      | localdatetime.transaction |
      | localdatetime.statement   |
      | localdatetime.realtime    |
      | datetime                  |
      | datetime.transaction      |
      | datetime.statement        |
      | datetime.realtime         |
      | duration                  |
