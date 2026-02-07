---
name: Use parameterized queries for cache reuse
description: Use CYPHER parameters in queries to enable query plan caching and improve performance
---

# Use parameterized queries for cache reuse

Use parameters in your queries to enable query plan caching and improve performance.

## Usage

Prefix your query with `CYPHER` followed by parameter declarations. Reference parameters with `$` syntax in the query body.

## Example

    redis-cli GRAPH.QUERY social "CYPHER name='Alice' MATCH (u:User {name: $name}) RETURN u.id"

## Notes

- Parameterized queries allow FalkorDB to cache and reuse query execution plans
- Parameters are declared after the `CYPHER` keyword using `name=value` syntax
- Parameters are referenced in the query using `$name` syntax
- This improves performance by avoiding repeated query parsing and planning
- Parameterized queries also provide better security by preventing injection attacks
