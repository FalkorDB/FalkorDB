from common import *


class testUDFCluster():
    def __init__(self):
        self.env, self.db = Env(env='oss-cluster', shardsCount=3)
        self.master_1 = self.env.getConnection(shardId=1)
        self.master_2 = self.env.getConnection(shardId=2)
        self.master_3 = self.env.getConnection(shardId=3)
        self.shards = [self.master_1, self.master_2, self.master_3]

    def tearDown(self):
        for shard in self.shards:
            shard.execute_command("FLUSHALL")
            shard.execute_command("GRAPH.UDF", "FLUSH")

    def test_udf_load_propagation(self):
        """
        load UDF to one master shard and make sure
        that on success the UDF is propagated to the rest of the cluster
        """

        # make sure all shards are clean
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(len(udfs), 0)

        # load UDF to master_1
        script = """
        falkor.register('add', function(a,b) {return a + b;});
        """

        res = self.db.udf_load("math", script)
        self.env.assertEqual(res, "OK")

        # collect UDFs from master_1
        master_1_udfs = self.master_1.execute_command("GRAPH.UDF", "LIST")
        self.env.assertNotEqual(len(master_1_udfs), 0)

        # make sure UDFs been propagated to the rest of the cluster
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_1_udfs, udfs)

        # update UDFs on master_2
        script = """
        falkor.register('add', function(a,b) {return a + b;});
        falkor.register('sub', function(a,b) {return a - b;});
        """

        res = self.db.udf_load("math", script, replace=True)
        self.env.assertEqual(res, "OK")

        # collect UDFs from master_2
        master_2_udfs = self.master_2.execute_command("GRAPH.UDF", "LIST")
        self.env.assertNotEqual(len(master_2_udfs), 0)

        # make sure UDFs been propagated to the rest of the cluster
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_2_udfs, udfs)

        # make sure a failed load doesn't effects the cluster
        try:
            # LOAD should fail as 'math' lib already exists and we did not
            # specified REPLACE
            self.db.udf_load("math", script)
            self.env.assertFalse(True)
        except Exception:
            pass

        # make sure UDFs remaind as before the failed call
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_2_udfs, udfs)

    def test_udf_delete_propagation(self):
        """
        delete UDF from one master shard and make sure
        that on success the UDF is deleted from the rest of the cluster
        """

        # load 3 libraries
        libs    = ["A", "B", "C"]
        scripts = ["falkor.register('a', function(a) {return a;});",
                   "falkor.register('b', function(b) {return b;});",
                   "falkor.register('c', function(c) {return c;});"]

        for i in range(0, 3):
            lib    = libs[i]
            script = scripts[i]

            res = self.db.udf_load(lib, script)
            self.env.assertEqual(res, "OK")

        # make sure all 3 libs are available throughout the cluster
        master_1_udfs = self.master_1.execute_command("GRAPH.UDF", "LIST")
        self.env.assertEqual(len(master_1_udfs), 3)

        # make sure UDFs been propagated to the rest of the cluster
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_1_udfs, udfs)

        # start removing libs
        remove_sequance = [(self.master_2, "B"),  # remove B from master 2
                           (self.master_3, "A"),  # remove A from master 3
                           (self.master_1, "C")]  # remove C from master 1

        for shard, lib in remove_sequance:
            res = self.db.udf_delete(lib)
            self.env.assertEqual(res, "OK")

            # make sure all nodes in the cluster has the same view over UDFs
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            for s in self.shards:
                s_udfs = s.execute_command("GRAPH.UDF", "LIST")
                self.env.assertEqual(udfs, s_udfs)

        # all shards should have no UDFs
        for s in self.shards:
            udfs = s.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(len(udfs), 0)

    def test_udf_flush_propagation(self):
        """
        flush UDFs from one master shard and make sure
        that on success the UDFs been flushed from the rest of the cluster
        """
        # load 3 libraries
        libs    = ["A", "B", "C"]
        scripts = ["falkor.register('a', function(a) {return a;});",
                   "falkor.register('b', function(b) {return b;});",
                   "falkor.register('c', function(c) {return c;});"]

        for i in range(0, 3):
            lib    = libs[i]
            script = scripts[i]

            res = self.db.udf_load(lib, script)
            self.env.assertEqual(res, "OK")

        # make sure all 3 libs are available throughout the cluster
        master_1_udfs = self.master_1.execute_command("GRAPH.UDF", "LIST")
        self.env.assertEqual(len(master_1_udfs), 3)

        # make sure UDFs been propagated to the rest of the cluster
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_1_udfs, udfs)

        # flush UDFs
        res = self.db.udf_flush()
        self.env.assertEqual(res, "OK")

        # all shards should have no UDFs
        for s in self.shards:
            udfs = s.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(len(udfs), 0)
