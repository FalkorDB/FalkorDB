from common import Env, FalkorDB, SANITIZER
import os
import tempfile
import shutil
import stat
import pytest


class testGraphCopyTempFolder:
    def teardown_method(self):
        self.conn.delete(self.src, self.dest)

    def set_temp_folder(self, path):
        module_args = f"TEMP_FOLDER {path}"
        self.env, self.db = Env(moduleArgs=module_args, enableDebugCommand=True)
        self.conn = self.env.getConnection()
        self.src = 'temp_src_graph'
        self.dest = 'temp_dest_graph'
        src_graph = self.db.select_graph(self.src)
        src_graph.query("RETURN 1")

    def test_01_temp_folder_is_file(self):
        if SANITIZER:
            self.env.skip()
        fd, file_path = tempfile.mkstemp()
        os.close(fd)
        try:
            with pytest.raises(Exception):
                self.set_temp_folder(file_path)
        finally:
            os.remove(file_path)

    def test_02_temp_folder_not_exist(self):
        if SANITIZER:
            self.env.skip()
        non_existent = "/tmp/falkordb_nonexistent_dir"
        if os.path.exists(non_existent):
            shutil.rmtree(non_existent)
        with pytest.raises(Exception):
            self.set_temp_folder(non_existent)

    def test_03_temp_folder_no_permission(self):
        if SANITIZER:
            self.env.skip()
        no_perm_dir = tempfile.mkdtemp()
        os.chmod(no_perm_dir, stat.S_IREAD)
        try:
            with pytest.raises(Exception):
                self.set_temp_folder(no_perm_dir)
        finally:
            os.chmod(no_perm_dir, stat.S_IWUSR | stat.S_IREAD | stat.S_IXUSR)
            shutil.rmtree(no_perm_dir)

    def test_04_temp_folder_exists_success(self):
        if SANITIZER:
            self.env.skip()
        valid_dir = tempfile.mkdtemp()
        self.set_temp_folder(valid_dir)
        try:
            self.conn.execute_command("GRAPH.COPY", self.src, self.dest)
            # Should succeed, check dest exists
            assert self.conn.type(self.dest) == 'graphdata'
        finally:
            shutil.rmtree(valid_dir)
            self.conn.delete(self.dest)
