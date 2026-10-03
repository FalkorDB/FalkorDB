# Testing the FalkorDB Snap

This document provides instructions for testing the FalkorDB snap package before and after publication.

## Local Testing

### Prerequisites

```bash
# Install snapcraft
sudo snap install snapcraft --classic

# Install LXD for building (if not already installed)
sudo snap install lxd
sudo lxd init --auto
```

### Build the Snap

From the repository root:

```bash
# Clean previous builds
snapcraft clean

# Build the snap
snapcraft --verbose
```

This will create a `falkordb_<version>_<arch>.snap` file in the current directory.

### Install and Test Locally

```bash
# Install the snap (dangerous flag allows local unsigned snaps)
sudo snap install --dangerous falkordb_*.snap

# Check service status
sudo snap services falkordb

# View logs
sudo snap logs falkordb -f
```

### Test Functionality

#### 1. Test Redis Connection

```bash
# Connect using the snap's CLI
falkordb.cli

# In redis-cli:
127.0.0.1:6379> PING
# Should return: PONG
```

#### 2. Test FalkorDB Module

```bash
# Connect with redis-cli
falkordb.cli

# Create a simple graph
127.0.0.1:6379> GRAPH.QUERY test "CREATE (:Person {name: 'Alice', age: 30})"

# Query the graph
127.0.0.1:6379> GRAPH.QUERY test "MATCH (p:Person) RETURN p.name, p.age"

# Delete the graph
127.0.0.1:6379> GRAPH.DELETE test
```

#### 3. Test Data Persistence

```bash
# Create some data
falkordb.cli
127.0.0.1:6379> GRAPH.QUERY persist "CREATE (:Test {value: 'persistent'})"
127.0.0.1:6379> QUIT

# Restart the service
sudo snap restart falkordb

# Verify data persists
falkordb.cli
127.0.0.1:6379> GRAPH.QUERY persist "MATCH (t:Test) RETURN t.value"
# Should return: persistent
```

#### 4. Test Service Management

```bash
# Stop the service
sudo snap stop falkordb

# Verify it's stopped
sudo snap services falkordb

# Start the service
sudo snap start falkordb

# Verify it's running
sudo snap services falkordb
```

### Cleanup

```bash
# Remove the snap
sudo snap remove falkordb

# Remove build artifacts
rm -f falkordb_*.snap
snapcraft clean
```

## Testing CI/CD Build

### Test PR Build (No Publishing)

1. Create a pull request with changes
2. Check the "Build and Publish Snap" workflow
3. Verify both amd64 and arm64 builds succeed
4. Download artifacts from the workflow run
5. Test the artifacts locally (see above)

### Test Edge Channel Build

1. Merge to master branch
2. Verify the workflow publishes to edge channel
3. Install from edge:
   ```bash
   sudo snap install falkordb --edge
   ```
4. Test functionality (see tests above)

### Test Stable Release

1. Create and push a version tag (e.g., `v4.2.1`)
2. Verify workflow publishes to stable and candidate channels
3. Install from stable:
   ```bash
   sudo snap install falkordb
   ```
4. Test functionality (see tests above)

## Automated Testing Script

Here's a complete test script:

```bash
#!/bin/bash
set -e

echo "=== FalkorDB Snap Test Suite ==="

# Test 1: Redis connectivity
echo "Test 1: Redis connectivity"
timeout 5 falkordb.cli PING || (echo "FAIL: Cannot connect to Redis" && exit 1)
echo "PASS"

# Test 2: FalkorDB module loaded
echo "Test 2: FalkorDB module loaded"
OUTPUT=$(falkordb.cli GRAPH.QUERY test "CREATE (:TestNode {id: 1})")
if [[ $OUTPUT == *"error"* ]]; then
    echo "FAIL: FalkorDB module not loaded"
    exit 1
fi
echo "PASS"

# Test 3: Query functionality
echo "Test 3: Query functionality"
falkordb.cli GRAPH.QUERY test "CREATE (:Person {name: 'Bob', age: 25})" > /dev/null
OUTPUT=$(falkordb.cli GRAPH.QUERY test "MATCH (p:Person) RETURN p.name" | grep "Bob")
if [[ -z $OUTPUT ]]; then
    echo "FAIL: Query did not return expected results"
    exit 1
fi
echo "PASS"

# Test 4: Data persistence
echo "Test 4: Data persistence"
falkordb.cli GRAPH.QUERY persist "CREATE (:Persistent {value: 'test'})" > /dev/null
sudo snap restart falkordb
sleep 5
OUTPUT=$(falkordb.cli GRAPH.QUERY persist "MATCH (p:Persistent) RETURN p.value" | grep "test")
if [[ -z $OUTPUT ]]; then
    echo "FAIL: Data did not persist after restart"
    exit 1
fi
echo "PASS"

# Cleanup
echo "Cleanup: Removing test data"
falkordb.cli GRAPH.DELETE test > /dev/null || true
falkordb.cli GRAPH.DELETE persist > /dev/null || true

echo "=== All tests passed! ==="
```

Save this as `test-snap.sh`, make it executable, and run:

```bash
chmod +x test-snap.sh
./test-snap.sh
```

## Architecture-Specific Testing

### Test on ARM64

If you have access to an ARM64 system:

```bash
# Build for ARM64
snapcraft --target-arch=arm64

# Install and test (same tests as above)
sudo snap install --dangerous falkordb_*_arm64.snap
```

### Test on AMD64

```bash
# Build for AMD64 (default)
snapcraft

# Install and test
sudo snap install --dangerous falkordb_*_amd64.snap
```

## Troubleshooting Tests

### Service Won't Start

```bash
# Check detailed logs
sudo snap logs falkordb --follow

# Check if port is available
sudo netstat -tulpn | grep 6379

# Try manual start with debug
sudo snap stop falkordb
sudo snap start falkordb
```

### Module Not Loading

```bash
# Verify module exists
ls -l /snap/falkordb/current/usr/local/lib/falkordb/

# Check Redis can load the module
sudo snap logs falkordb | grep -i "module.*load"
```

### Permission Issues

```bash
# Check snap permissions
snap connections falkordb

# Verify data directory
ls -la /var/snap/falkordb/common/
```

## Reporting Issues

If you encounter issues during testing:

1. Collect logs:
   ```bash
   sudo snap logs falkordb > falkordb-logs.txt
   ```

2. Include system information:
   ```bash
   snap --version
   uname -a
   ```

3. Report at: https://github.com/FalkorDB/FalkorDB/issues
