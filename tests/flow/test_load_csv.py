import csv
from common import *

GRAPH_ID = "load_csv"

EMPTY_CSV                               = "empty.csv"
EMPTY_CSV_RELATIVE_PATH                 = "./tests/flow/" + EMPTY_CSV
EMPTY_CSV_HEADER                        = []
EMPTY_CSV_DATA                          = []

SHORT_CSV_WITH_HEADERS                  = "short_with_header.csv"
SHORT_CSV_WITH_HEADERS_RELATIVE_PATH    = "./tests/flow/" + SHORT_CSV_WITH_HEADERS
SHORT_CSV_WITH_HEADERS_HEADER           = [["First Name", "Last Name"]]
SHORT_CSV_WITH_HEADERS_DATA             = [["Roi",  "Lipman"],
                                           ["Hila", "Lipman"],
                                           ["Adam", "Lipman"],
                                           ["Yoav", "Lipman"]]

SHORT_CSV_WITHOUT_HEADERS               = "short_without_header.csv"
SHORT_CSV_WITHOUT_HEADERS_RELATIVE_PATH = "./tests/flow/" + SHORT_CSV_WITHOUT_HEADERS
SHORT_CSV_WITHOUT_HEADERS_HEADER        = []
SHORT_CSV_WITHOUT_HEADERS_DATA          = [["Roi",  "Lipman"],
                                           ["Hila", "Lipman"],
                                           ["Adam", "Lipman"],
                                           ["Yoav", "Lipman"]]

# write a CSV file using 'name' as the file name
# 'header' [optional] as the first row
# 'data' [optional] CSV rows
def create_csv_file(name, header, data, delimiter=','):
    with open(name, 'w') as f:
        writer = csv.writer(f)
        data = header + data
        writer.writerows(data)

# create an empty CSV file
def create_empty_csv():
    name   = EMPTY_CSV
    DATA   = EMPTY_CSV_DATA
    HEADER = EMPTY_CSV_HEADER
    create_csv_file(name, HEADER, DATA)

    return name

# create a short CSV file with a header row
def create_short_csv_with_header():
    name   = SHORT_CSV_WITH_HEADERS
    DATA   = SHORT_CSV_WITH_HEADERS_DATA
    HEADER = SHORT_CSV_WITH_HEADERS_HEADER

    create_csv_file(name, HEADER, DATA)

    return name

# create a short CSV file without a header row
def create_short_csv_without_header():
    name   = SHORT_CSV_WITHOUT_HEADERS 
    DATA   = SHORT_CSV_WITHOUT_HEADERS_DATA
    HEADER = SHORT_CSV_WITHOUT_HEADERS_HEADER

    create_csv_file(name, HEADER, DATA)

    return name

class testLoadCSV():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

        # create CSV files
        create_empty_csv()
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

    def test02_project_csv_rows(self):
        g = self.graph

        # project all rows in a CSV file
        q = """LOAD CSV FROM $file AS row
               RETURN row"""

        datasets = [(EMPTY_CSV_RELATIVE_PATH,                 []),
                    (SHORT_CSV_WITH_HEADERS_RELATIVE_PATH,    [*SHORT_CSV_WITH_HEADERS_HEADER,    *SHORT_CSV_WITH_HEADERS_DATA]),
                    (SHORT_CSV_WITHOUT_HEADERS_RELATIVE_PATH, [*SHORT_CSV_WITHOUT_HEADERS_HEADER, *SHORT_CSV_WITHOUT_HEADERS_DATA])]

        for dataset in datasets:
            # project all rows from CSV file
            file_name = dataset[0]
            expected  = dataset[1]

            result = g.query(q, {'file': file_name}).result_set
            for i, row in enumerate(result):
                # validate result
                self.env.assertEquals(row[0], expected[i])

    def test03_project_csv_as_map(self):
        g = self.graph

        # project all rows in a CSV file
        q = """LOAD CSV WITH HEADERS FROM $file AS row
                RETURN row"""

        datasets = [(EMPTY_CSV_RELATIVE_PATH,                 EMPTY_CSV_HEADER,                  EMPTY_CSV_DATA),
                    (SHORT_CSV_WITHOUT_HEADERS_RELATIVE_PATH, SHORT_CSV_WITHOUT_HEADERS_DATA[0], SHORT_CSV_WITHOUT_HEADERS_DATA[1:]),
                    (SHORT_CSV_WITH_HEADERS_RELATIVE_PATH,    SHORT_CSV_WITH_HEADERS_HEADER[0],  SHORT_CSV_WITH_HEADERS_DATA)]

        for dataset in datasets:
            file    = dataset[0]
            columns = dataset[1]
            data    = dataset[2]

            expected = []
            for row in data:
                obj = {}
                for idx, column in enumerate(columns):
                    obj[column] = row[idx]
                expected.append(obj)
            
            result = g.query(q, {'file': file}).result_set
            self.env.assertEquals(result, expected)

    def _test04_load_csv_multiple_times(self):
        # project the same CSV multiple times
        q = """UNWIND range(0, 3) AS x
               LOAD CSV FROM $file AS row
               RETURN x, row
               ORDER BY x"""

        result = self.graph.query(q, {'file': SHORT_CSV_WITHOUT_HEADERS}).result_set

        expected = []
        for i in range(3):
            for row in SHORT_CSV_WITHOUT_HEADERS_DATA:
                expected.append([i, row])

        self.env.assertEquals(result, expected)

    def _test05_load_multiple_files(self):
        g = self.graph

        # project multiple CSV files
        q = """LOAD CSV FROM $file_1 AS row
               WITH collect(row) as file_1_rows
               LOAD CSV FROM $file_2 AS row
               RETURN file_1_rows, collect(row) as file_2_rows
               """
        result = g.query(q, {'file_1': SHORT_CSV_WITHOUT_HEADERS,
                             'file_2': SHORT_CSV_WITH_HEADERS}).result_set

        file_1_rows = SHORT_CSV_WITHOUT_HEADERS_DATA
        file_2_rows = [SHORT_CSV_WITH_HEADERS_HEADER + SHORT_CSV_WITH_HEADERS_DATA]

        self.env.assertEquals(result[0], file_1_rows)
        self.env.assertEquals(result[1], file_2_rows)
