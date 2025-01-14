import os
import csv
from common import *
from pathlib import Path
from collections import OrderedDict

GRAPH_ID_LOCAL = "local_load_csv"
GRAPH_ID_REMOTE = "remote_load_csv"

EMPTY_CSV                               = "empty.csv"
EMPTY_CSV_HEADER                        = []
EMPTY_CSV_DATA                          = []

SHORT_CSV_WITH_HEADERS                  = "short_with_header.csv"
SHORT_CSV_WITH_HEADERS_HEADER           = [["First Name", "Last Name"]]
SHORT_CSV_WITH_HEADERS_DATA             = [["Adam", "Lipman"],
                                           ["Hila", "Lipman"],
                                           ["Roi",  "Lipman"],
                                           ["Yoav", "Lipman"]]

SHORT_CSV_WITHOUT_HEADERS               = "short_without_header.csv"
SHORT_CSV_WITHOUT_HEADERS_HEADER        = []
SHORT_CSV_WITHOUT_HEADERS_DATA          = [["Adam", "Lipman"],
                                           ["Hila", "Lipman"],
                                           ["Roi",  "Lipman"],
                                           ["Yoav", "Lipman"]]

MALFORMED_CSV                           = "malformed.csv"
MALFORMED_CSV_HEADER                    = [["FirstName", "LastName"]]
MALFORMED_CSV_DATA                      = [["Roi",  "Lipman"],
                                           ["Yoav", "Lipman", "Extra"]]

EMPTY_CELL_CSV                          = "empty_cell.csv"
EMPTY_CELL_CSV_HEADER                   = ["FirstName", "LastName", "Age"]

EMPTY_COLUMN_CSV                        = "empty_column.csv"

IMPORT_DIR = None

# write a CSV file using 'name' as the file name
# 'header' [optional] as the first row
# 'data' [optional] CSV rows
def create_csv_file(name, header, data, delimiter=','):
    # create any missing directories in the path
    path = IMPORT_DIR + name
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        writer = csv.writer(f)
        data = header + data
        writer.writerows(data)

# create an empty CSV file
def create_empty_csv():
    name   = EMPTY_CSV
    data   = EMPTY_CSV_DATA
    header = EMPTY_CSV_HEADER
    create_csv_file(name, header, data)

# create a short CSV file with a header row
def create_short_csv_with_header():
    name   = SHORT_CSV_WITH_HEADERS
    data   = SHORT_CSV_WITH_HEADERS_DATA
    header = SHORT_CSV_WITH_HEADERS_HEADER

    create_csv_file(name, header, data)

# create a short CSV file without a header row
def create_short_csv_without_header():
    name   = SHORT_CSV_WITHOUT_HEADERS 
    data   = SHORT_CSV_WITHOUT_HEADERS_DATA
    header = SHORT_CSV_WITHOUT_HEADERS_HEADER

    create_csv_file(name, header, data)

# create a malformed CSV file
def create_malformed_csv():
    name   = MALFORMED_CSV
    data   = MALFORMED_CSV_DATA
    header = MALFORMED_CSV_HEADER

    create_csv_file(name, header, data)

# create a CSV with an empty cell
def create_empty_cell_csv():
    # create any missing directories in the path
    name = EMPTY_CELL_CSV
    path = IMPORT_DIR + name
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        header = ",".join(EMPTY_CELL_CSV_HEADER)
        f.write(f"{header}\n")
        f.write("roi,,40\n")

# create a CSV with an empty column
def create_empty_column_csv():
    # create any missing directories in the path
    name = EMPTY_COLUMN_CSV
    path = IMPORT_DIR + name
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        header = "fist_column,,third_column"
        f.write(f"{header}\n")
        f.write("A,B,C\nD,E,F")

class testLoadLocalCSV():
    def __init__(self):
        # Get the absolute path to the current file
        global IMPORT_DIR
        IMPORT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

        self.env, self.db = Env(moduleArgs=f"IMPORT_FOLDER {IMPORT_DIR}")
        self.graph = self.db.select_graph(GRAPH_ID_LOCAL)

        # create CSV files
        create_empty_csv()
        create_malformed_csv()
        create_empty_cell_csv()
        create_empty_column_csv()
        create_short_csv_with_header()
        create_short_csv_without_header()
 
    # test invalid invocations of the LOAD CSV command
    def test01_invalid_call(self):
        queries = ["LOAD CSV FROM a AS row RETURN row",
                   "LOAD CSV WITH HEADERS FROM a AS row RETURN row",

                   "LOAD CSV FROM 2 AS row RETURN row",
                   "LOAD CSV WITH HEADERS FROM 2 AS row RETURN row",

                   "LOAD CSV FROM $arr AS row RETURN row",
                   "LOAD CSV WITH HEADERS FROM $arr AS row RETURN row",

                   "WITH 2 AS x LOAD CSV FROM x AS row RETURN row",
                   "WITH 2 AS x LOAD CSV WITH HEADERS FROM x AS row RETURN row"
                   ]

        for q in queries:
            try:
                self.graph.query(q, {'arr': []})
                self.env.assertFalse(True)
            except Exception as e:
                continue

    def test02_none_existing_csv_file(self):
        q = "LOAD CSV FROM 'file://none_existing.csv' AS row RETURN row"
        try:
            self.graph.query(q)
            self.env.assertFalse(True)
        except Exception as e:
            # failed to open CSV file: a
            pass


    def test03_none_supported_uri(self):
        URIS = ["http", "ftp", "ssh", "telnet"]
        for uri in URIS:
            q = f"LOAD CSV FROM '{uri}://none_existing.csv' AS row RETURN row"
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except Exception as e:
                # failed to open CSV file: a
                pass

    def test04_malformed_csv(self):
        queries = ["LOAD CSV FROM $file AS row RETURN row",
                   "LOAD CSV WITH HEADERS FROM $file AS row RETURN row"]

        for q in queries:
            try:
                self.graph.query(q, {'file': 'file://' + MALFORMED_CSV})
                self.env.assertFalse(True)
            except Exception as e:
                # failed to process malformed csv
                pass

    def test05_empty_cell_csv(self):
        q = "LOAD CSV FROM $file AS row RETURN row ORDER BY row"
        result = self.graph.query(q, {'file': 'file://' + EMPTY_CELL_CSV}).result_set
        actual = result[1][0] # skip header row
        self.env.assertIn("roi", actual)
        self.env.assertIn(None, actual)

        q = "LOAD CSV WITH HEADERS FROM $file AS row RETURN row"
        result = self.graph.query(q, {'file': 'file://' + EMPTY_CELL_CSV}).result_set
        actual = result[0][0]
        self.env.assertNotIn('LastName', actual)
        self.env.assertEquals(actual['FirstName'], 'roi')
        self.env.assertEquals(actual['Age'], '40')

        # create nodes from empty cell csv
        q = "LOAD CSV WITH HEADERS FROM $file AS row CREATE (p:Person) SET p = row RETURN p"
        result = self.graph.query(q, {'file': 'file://' + EMPTY_CELL_CSV}).result_set
        node = result[0][0]
        self.env.assertEqual(len(node.properties), 2)
        self.env.assertEquals(node.properties['Age'], '40')
        self.env.assertEquals(node.properties['FirstName'], 'roi')

    def test06_empty_column_csv(self):
        q = "LOAD CSV WITH HEADERS FROM $file AS row RETURN row ORDER BY row"
        try:
            self.graph.query(q, {'file': 'file://' + EMPTY_COLUMN_CSV})
            # CSV empty column name
            self.env.assertFalse("we should have failed" and False)
        except Exception:
            pass

    def test07_project_csv_rows(self):
        g = self.graph

        # project all rows in a CSV file
        q = """LOAD CSV FROM $file AS row
               RETURN row
               ORDER BY row"""

        datasets = [(EMPTY_CSV,                 []),
                    (SHORT_CSV_WITH_HEADERS,    [*SHORT_CSV_WITH_HEADERS_HEADER,    *SHORT_CSV_WITH_HEADERS_DATA]),
                    (SHORT_CSV_WITHOUT_HEADERS, [*SHORT_CSV_WITHOUT_HEADERS_HEADER, *SHORT_CSV_WITHOUT_HEADERS_DATA])]

        for dataset in datasets:
            # project all rows from CSV file
            file_name = dataset[0]
            expected  = dataset[1]
            expected = sorted(expected)

            result = g.query(q, {'file': 'file://' + file_name}).result_set
            for i, row in enumerate(result):
                # validate result
                self.env.assertEquals(row[0], expected[i])

    def test08_project_csv_as_map(self):
        g = self.graph

        # project all rows in a CSV file
        q = """LOAD CSV WITH HEADERS FROM $file AS row
                RETURN row
                ORDER BY row"""

        datasets = [(SHORT_CSV_WITHOUT_HEADERS, SHORT_CSV_WITHOUT_HEADERS_DATA[0], SHORT_CSV_WITHOUT_HEADERS_DATA[1:]),
                    (SHORT_CSV_WITH_HEADERS,    SHORT_CSV_WITH_HEADERS_HEADER[0],  SHORT_CSV_WITH_HEADERS_DATA)]

        for dataset in datasets:
            file    = dataset[0]
            columns = dataset[1]
            data    = dataset[2]

            expected = []
            for row in data:
                obj = {}
                for idx, column in enumerate(columns):
                    obj[column] = row[idx]
                expected.append([obj])
            
            result = g.query(q, {'file': 'file://' + file}).result_set
            self.env.assertEquals(result, expected)

    def test09_load_csv_multiple_times(self):
        # project the same CSV multiple times
        q = """UNWIND range(0, 3) AS x
               LOAD CSV FROM $file AS row
               RETURN x, row
               ORDER BY x, row"""

        result = self.graph.query(q, {'file': 'file://' + SHORT_CSV_WITHOUT_HEADERS}).result_set

        expected = []
        for i in range(4):
            for row in SHORT_CSV_WITHOUT_HEADERS_DATA:
                expected.append([i, row])

        self.env.assertEquals(result, expected)

    def test10_load_multiple_files(self):
        g = self.graph

        # project multiple CSV files
        q = """LOAD CSV FROM $file_1 AS row
               WITH collect(row) as file_1_rows
               LOAD CSV FROM $file_2 AS row
               RETURN file_1_rows, collect(row) as file_2_rows
               """
        result = g.query(q, {'file_1': 'file://' + SHORT_CSV_WITHOUT_HEADERS,
                             'file_2': 'file://' + SHORT_CSV_WITH_HEADERS}).result_set

        file_1_rows = SHORT_CSV_WITHOUT_HEADERS_DATA
        file_2_rows = SHORT_CSV_WITH_HEADERS_HEADER + SHORT_CSV_WITH_HEADERS_DATA

        self.env.assertEquals(len(file_1_rows), len(result[0][0]))
        for item in file_1_rows:
            self.env.assertTrue(item in result[0][0])

        self.env.assertEquals(len(file_2_rows), len(result[0][1]))
        for item in file_2_rows:
            self.env.assertTrue(item in result[0][1])

    def test11_breakout_import_folder(self):
        # try accessing files outside of the import directory
        g = self.graph

        # try accessing the hosts file
        # default import path: /var/lib/FalkorDB/import/
        q = """LOAD CSV FROM 'file://../../../etc/hosts AS row
               RETURN row"""

        try:
            res = g.query(q).result_set
            self.env.assertFalse("we should not be here" and False)
        except:
            pass

class testLoadRemoteCSV():
    def __init__(self):
        self.env, self.db = Env(moduleArgs=f"IMPORT_FOLDER {IMPORT_DIR}")

        # skip test if we're running under Valgrind
        if VALGRIND or SANITIZER != "":
            self.env.skip() # libcrypto.so seems to crash when running under sanitizer

        self.graph = self.db.select_graph(GRAPH_ID_REMOTE)

    # test invalid invocations of the LOAD CSV command
    def test01_load_remote_csv(self):
        query = "LOAD CSV FROM $url AS row RETURN row"
        url = "https://raw.githubusercontent.com/FalkorDB/FalkorDB/refs/heads/master/demo/social/resources/friends.csv"

        # get the directory of the current file
        current_folder_path = Path(__file__).parent

        # build the path to the target file
        target_file_path = current_folder_path / ".." / ".." / "demo" / "social" / "resources" / "friends.csv"
        target_file_path = target_file_path.resolve()  # Convert to absolute path

        # Read the CSV file into a list of lists
        data = []
        with open(target_file_path, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            data = [row for row in csv_reader]  # Convert rows to a list of lists

        result = self.graph.query(query, {'url': url}).result_set
        for row in result:
            self.env.assertIn(row[0], data)

    def test_02_none_existing_url(self):
        query = "LOAD CSV FROM $url AS row RETURN row"
        urls = ["https://fakljsmndklnmsdvnkndqw02emkl.dodndasno12.dal/text.csv"]

        for url in urls:
            try:
                self.graph.query(query, {'url': url})
                self.env.assertFalse("we should have failed" and False)
            except Exception:
                pass

