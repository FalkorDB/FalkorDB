# Crash Issues Review - November 2025

This document provides a comprehensive review of all crash-related issues in the FalkorDB repository.

## Executive Summary

- **Total crash-related issues found**: 107
- **Currently open**: 26 issues
- **High priority**: 10 issues require immediate attention
- **Status**: Most recent crashes confirmed in November 2025

## Verification Status

### ✅ Confirmed Active Crashes (2025)

#### 1. Issue #1353 - OPTIONAL MATCH with Multiple Path Variables
- **Created**: November 8, 2025
- **Status**: Open, Assigned to Copilot
- **Severity**: High
- **Reproduction Query**:
  ```cypher
  OPTIONAL MATCH (n0)--(n1)
  OPTIONAL MATCH p0 = (n1), p1 = (n0)
  RETURN *
  ```
- **Impact**: Query crashes database
- **Next Steps**: Needs investigation and fix

#### 2. Issue #1340 - Unique Property with Space Characters
- **Created**: November 4, 2025
- **Status**: Open, Assigned to @swilly22
- **Severity**: Critical
- **Reproduction Query**:
  ```cypher
  CREATE INDEX FOR (c:Customer) ON (c.ref)
  GRAPH.CONSTRAINT CREATE tg4 UNIQUE NODE Customer PROPERTIES 1 ref
  MERGE (c:Customer {ref:'1093436713'})
  MERGE (c:Customer {ref:' 1093436713'})  # Crashes here
  ```
- **Impact**: Database crashes when unique property starts or ends with space
- **Next Steps**: Fix unique constraint handling

#### 3. Issue #415 - GraphEntity_Keys Crash
- **Created**: September 25, 2023
- **Status**: Open, Still Reproducible (confirmed Nov 9, 2025)
- **Severity**: High
- **Version**: Still crashes in v4.14.5
- **Reproduction Query**:
  ```cypher
  CREATE (root:Root {name: 'x'}), 
         (child1:TextNode {var: floor(any(v4 IN [2] WHERE child1 = [root IN keys(root)]))})
  ```
- **Impact**: Accessing keys() in nested expressions crashes
- **Reward**: $100 bounty available
- **Next Steps**: Long-standing issue needs fix

#### 4. Issue #636 - Fuzzer-Found Crash
- **Created**: April 15, 2024
- **Status**: Open, Active PR #1289
- **Severity**: High
- **Simplified Reproduction** (by @swilly22):
  ```cypher
  CREATE (:A), (:B)<-[:R0]-()<-[:R1]-()
  MATCH (n:A)<-[*]-(n:Z) RETURN 1
  ```
- **Impact**: Variable-length pattern matching crash
- **Reward**: $100 bounty available
- **Next Steps**: PR in review, needs completion

#### 5. Issue #1240 - Multiple Connections Crash
- **Created**: August 25, 2025
- **Status**: Open, Assigned to @swilly22
- **Severity**: Critical (Production)
- **Conditions**:
  - Multiple concurrent connections (2 or more)
  - Write queries present
  - Occurs regardless of connection pool size
  - Dataset size ~40MB minimum
- **Impact**: Production environments with multiple users crash
- **Next Steps**: Concurrency bug needs fix

#### 6. Issue #910 - Redis Replicas Pod Crash
- **Created**: January 15, 2025
- **Status**: Open, Assigned to @AviAvni
- **Severity**: Critical (Production)
- **Error**: `AttributeSet_AddNoClone` crash during AOF loading
- **Impact**: Replica nodes crash during startup, breaking HA deployments
- **Next Steps**: Fix replication data loading

### 🔍 Needs Verification (Recent Reports)

#### 7. Issue #1333 - Multiple WHERE Clauses Crash
- **Created**: November 2, 2025
- **Status**: Open, Assigned to @swilly22
- **Needs**: Manual testing on latest build

#### 8. Issue #1336 - DETACH DELETE endNode Crash  
- **Created**: November 2, 2025
- **Status**: Open, Assigned to @swilly22
- **Note**: Works in Neo4j
- **Needs**: Manual testing on latest build

#### 9. Issue #807 - UNION ALL Crash
- **Created**: October 14, 2024
- **Status**: Open, Assigned to Copilot
- **Workaround**: Add LIMIT to subqueries
- **Needs**: Verification if still occurs

#### 10. Issue #1204 - RESTORE Command Crash
- **Created**: August 5, 2025
- **Status**: Open (Reopened), Assigned to @swilly22
- **Impact**: Backup/restore operations fail
- **Needs**: Verification on latest build

### ⚠️ May Need Investigation

The following issues are older but may still be valid:
- #751 - Self-connecting relation crash (Jul 2024)
- #753 - AlgebraicExpression_Dest crash (Jul 2024)
- #897 - Bolt protocol disconnect crash (Jan 2025)
- #955 - Telemetry crash (Feb 2025)
- #1158 - 358th query crash (Jun 2025)
- #1280 - WITH/MATCH crash (Sep 2025)
- #1281 - OPTIONAL MATCH crash (Sep 2025)
- #1300 - Filtered alias issue (Oct 2025)
- #1312 - Signal 8 on large graph (Oct 2025)

### 📚 Legacy RedisGraph Issues

Approximately 30-40 issues were migrated from RedisGraph. These need systematic verification:
- Many may have been fixed during the FalkorDB fork
- Some may no longer be relevant
- Priority should be given to recently confirmed issues

## Testing Strategy

### Immediate Actions (This Week)
1. ✅ Compiled list of all crash issues
2. ⏳ Create automated test cases for top 10 crashes
3. ⏳ Test each crash on latest main branch
4. ⏳ Update each issue with current status
5. ⏳ Close issues that are no longer reproducible

### Short Term (This Month)
1. Fix confirmed high-priority crashes (#1353, #1340, #415, #1240, #910)
2. Complete PR review for #636
3. Add regression tests for all fixed crashes
4. Implement crash detection in CI/CD

### Long Term (Next Quarter)
1. Systematic verification of all legacy issues
2. Close or fix remaining open crash issues
3. Add fuzzing to CI/CD pipeline
4. Improve error handling for edge cases
5. Add concurrency stress testing

## How to Test a Crash Issue

### Prerequisites
```bash
# Build FalkorDB
cd /home/runner/work/FalkorDB/FalkorDB
make

# Start Redis with FalkorDB
redis-server --loadmodule ./bin/linux-x64-release/src/falkordb.so
```

### Testing Procedure
1. Connect to Redis: `redis-cli`
2. Create a fresh graph: `GRAPH.QUERY test "RETURN 1"`
3. Execute the reproduction query
4. Observe:
   - Does it crash?
   - What error appears?
   - Check Redis logs
5. Document findings in the issue

### Example Test Case
```bash
# Test Issue #1353
redis-cli GRAPH.QUERY test "OPTIONAL MATCH (n0)--(n1) OPTIONAL MATCH p0 = (n1), p1 = (n0) RETURN *"
# Expected: Should not crash
# Actual: [Document what happens]
```

## Priority Matrix

| Priority | Issue | Type | Impact | Effort |
|----------|-------|------|--------|--------|
| P0 | #1240 | Concurrency | Production | High |
| P0 | #910 | Replication | Production | High |
| P0 | #1340 | Data Integrity | Production | Medium |
| P1 | #1353 | Query Processing | Functionality | Medium |
| P1 | #415 | Expression Eval | Functionality | Medium |
| P1 | #636 | Pattern Matching | Functionality | Medium |
| P2 | #1333 | WHERE Clause | Functionality | Low |
| P2 | #1336 | DELETE | Functionality | Low |
| P2 | #807 | UNION | Functionality | Low |
| P2 | #1204 | Backup/Restore | Operations | Medium |

## Recommendations

### For Maintainers
1. **Prioritize P0 issues** - These affect production deployments
2. **Create crash regression suite** - Prevent fixed crashes from reoccurring
3. **Add memory safety tools** - Run ASAN/MSAN in CI
4. **Improve error handling** - Convert crashes to error messages where possible
5. **Close stale issues** - Systematically verify old issues

### For Contributors
1. **Test before claiming** - Verify the crash is still reproducible
2. **Include tests** - Add regression test with your fix
3. **Document workarounds** - If a workaround exists, document it
4. **Ask for help** - Many of these are complex bugs

### For Users
1. **Report crashes** - Include query, data, and logs
2. **Check for duplicates** - Search existing issues first
3. **Test workarounds** - Some crashes have workarounds
4. **Stay updated** - Watch issues you're affected by

## Statistics

- **2025 Issues**: 15 crash issues reported
- **2024 Issues**: ~20 crash issues reported  
- **Pre-2024 Issues**: ~72 legacy issues
- **With PRs**: 1 (#636)
- **With Bounties**: 2 (#415, #636)
- **Confirmed Fixed**: Unknown (needs verification)

## Related Documents
- [Issue List](https://github.com/FalkorDB/FalkorDB/issues?q=is%3Aissue+crash)
- [Contributing Guide](../CONTRIBUTING.md)
- [Testing Guide](../tests/README.md)

## Last Updated
November 9, 2025

## Maintainers
- @gkorland
- @swilly22
- @AviAvni

---

*This is a living document. Please update it as issues are verified, fixed, or closed.*
