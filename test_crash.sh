#!/bin/bash
redis-cli FLUSHALL
redis-cli GRAPH.QUERY test "MERGE (:label8) MERGE (:label2{})<-[:reltype5]-(node_0{})<-[:reltype7]-({})"
redis-cli GRAPH.QUERY test "MATCH (node_0:label8{})<-[*..]-(node_0:label9) WHERE node_0.prop7 = [ FALSE ] RETURN *"
