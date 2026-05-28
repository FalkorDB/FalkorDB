#!/bin/bash

# WARNING: This script uses FLUSHALL which deletes all Redis data.
# Ensure you're running against a test instance only.

# Check if TEST_REDIS env var is set (safety guard)
if [ "${TEST_REDIS}" != "true" ]; then
  echo "Error: TEST_REDIS=true must be set to run this script (safety measure)."
  echo "Usage: TEST_REDIS=true bash test_crash.sh"
  exit 1
fi

# Check if Redis is accessible
if ! redis-cli ping > /dev/null 2>&1; then
  echo "Error: Redis is not accessible or not running."
  exit 1
fi

echo "Running crash test queries..."

redis-cli FLUSHALL
if [ $? -ne 0 ]; then
  echo "Error: FLUSHALL failed"
  exit 1
fi

redis-cli GRAPH.QUERY test "MERGE (:label8) MERGE (:label2{})<-[:reltype5]-(node_0{})<-[:reltype7]-({})"
if [ $? -ne 0 ]; then
  echo "Warning: First GRAPH.QUERY failed (exit code $?)"
fi

redis-cli GRAPH.QUERY test "MATCH (node_0:label8{})<-[*..]-(node_0:label9) WHERE node_0.prop7 = [ FALSE ] RETURN *"
if [ $? -ne 0 ]; then
  echo "Warning: Second GRAPH.QUERY failed (exit code $?)"
fi

echo "Crash test completed."
