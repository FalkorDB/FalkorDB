from common import *

GRAPH_ID = "parser"

class testParser(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()
    
    def test_multiple_single_line_comments(self):
        q = """
        MATCH (n:N)
        WHERE n.name IN ['A', 'B', 'C' ]
        WITH n AS n, collect(n.v) AS vs
        // first comment
        CALL {
            WITH n, vs
            // second comment
            WITH n AS n, vs AS vs, n.z + n.k AS zk
            UNWIND vs AS v
            RETURN
            count (DISTINCT n.x) AS x
        }

        // Third comment
        RETURN
            1 AS one,
            2 AS two
        ORDER BY x
        LIMIT 100

        // Fourth comment
        UNION ALL

        MATCH (n:N)
        WHERE n.name IN ['A', 'B', 'C' ]
        WITH n AS n, collect(n.v) AS vs
        CALL {
            WITH n, vs
            WITH n AS n, vs AS vs, n.z + n.k AS zk
            UNWIND vs AS v
            RETURN
            count (DISTINCT n.x) AS x
        }

        RETURN
            1 AS one,
            2 AS two
        ORDER BY x
        LIMIT 100
        """
        self.graph.query(q)

        # make sure graph.query ran without raising exception
        self.env.assertTrue(True)

    def test_multiple_multi_line_comments(self):
        q = """
        MATCH (n:N)
        WHERE n.name IN ['A', 'B', 'C' ]
        WITH n AS n, collect(n.v) AS vs
        /* first comment */
        CALL {
            WITH n, vs
            /* second comment

               continue
            */
            WITH n AS n, vs AS vs, n.z + n.k AS zk
            UNWIND vs AS v
            RETURN
            count (DISTINCT n.x) AS x
        }

        /* Third comment   */
        RETURN
            1 AS one,
            2 AS two
        ORDER BY x
        LIMIT 100

        /*Fourth comment*/
        UNION ALL

        MATCH (n:N)
        WHERE n.name IN ['A', 'B', 'C' ]
        WITH n AS n, collect(n.v) AS vs
        CALL {
            WITH n, vs
            WITH n AS n, vs AS vs, n.z + n.k AS zk
            UNWIND vs AS v
            RETURN
            count (DISTINCT n.x) AS x
        }

        RETURN
            1 AS one,
            2 AS two
        ORDER BY x
        LIMIT 100
        """
        self.graph.query(q)

        # make sure graph.query ran without raising exception
        self.env.assertTrue(True)

    def test_complex(self):
        q = """
        MATCH (b:B)
WHERE b.rn IN [
  'A','Ai','Cl','Co','Cod','K',
  's','sr'
]
MATCH (b)-[:H]->(c:C)
WHERE c.t >= '2026-01-01' AND c.t <= '2026-02-19'
WITH b.rn AS ro, collect(c) AS cs

// First comment
CALL {
  WITH ro, cs
  UNWIND cs AS cc
  RETURN
    count(DISTINCT cc.h)            AS tc,
    coalesce(sum(cc.la), 0)         AS tla,
    coalesce(sum(cc.ld), 0)         AS tld,
    coalesce(sum(CASE WHEN cc.im = true THEN 1 ELSE 0 END), 0) AS cmt
}

CALL {
  WITH ro, cs
  UNWIND cs AS cc
  WITH ro, substring(cc.t,0,10) AS day, count(DISTINCT cc.h) AS cod
  ORDER BY day
  RETURN collect({day: day, count: cod}) AS cpd
}

CALL {
  WITH ro, cs
  UNWIND cs AS cc
  WITH ro,
       substring(cc.t,0,4) AS year,
       toInteger(substring(cc.t,5,2)) AS mo,
       toInteger(substring(cc.t,8,2)) AS di,
       cc
  // second comment
  WITH ro, year, di +
    CASE mo
      WHEN 1 THEN 0 WHEN 2 THEN 31 WHEN 3 THEN 59 WHEN 4 THEN 90 WHEN 5 THEN 120 WHEN 6 THEN 151
      WHEN 7 THEN 181 WHEN 8 THEN 212 WHEN 9 THEN 243 WHEN 10 THEN 273 WHEN 11 THEN 304 WHEN 12 THEN 334 END
    AS doy, cc
  WITH ro, year, toInteger((doy - 1) / 7) + 1 AS wn, cc
  WITH ro, year + '-W' + toString(wn) AS wl, count(DISTINCT cc.h) AS ciw
  ORDER BY wl
  RETURN collect({week: wl, count: ciw}) AS cpw
}

CALL {
  WITH ro, cs
  UNWIND cs AS cc
  WITH ro, substring(cc.t,0,10) AS day, count(DISTINCT cc.h) AS cod
  ORDER BY cod DESC, day ASC
  LIMIT 5
  RETURN collect({day: day, count: cod}) AS tds
}

CALL {
  WITH ro, cs
  UNWIND cs AS cc
  WITH ro,
       substring(cc.t,0,4) AS year,
       toInteger(substring(cc.t,5,2)) AS mo,
       toInteger(substring(cc.t,8,2)) AS di,
       cc
  WITH ro, year, di +
    CASE mo
      WHEN 1 THEN 0 WHEN 2 THEN 31 WHEN 3 THEN 59 WHEN 4 THEN 90 WHEN 5 THEN 120 WHEN 6 THEN 151
      WHEN 7 THEN 181 WHEN 8 THEN 212 WHEN 9 THEN 243 WHEN 10 THEN 273 WHEN 11 THEN 304 WHEN 12 THEN 334 END
    AS doy, cc
  WITH ro, year, toInteger((doy - 1) / 7) + 1 AS wn, cc
  WITH ro, year + '-W' + toString(wn) AS wl, count(DISTINCT cc.h) AS ciw
  ORDER BY ciw DESC, wl ASC
  LIMIT 5
  RETURN collect({week: wl, count: ciw}) AS tw
}

// Third comment
RETURN
  ro                                   AS rn,
  tc,
  cpw                                  AS cpw,     // some comment of {a, b}
  cpd                                  AS cpd,     // diff comment {b, a}
  cmt,
  CASE WHEN tc = 0 THEN 0 ELSE toFloat(cmt) / toFloat(tc) END AS mrt,
  tla,
  tld,
  CASE WHEN tc = 0 THEN 0 ELSE toFloat(tla) / toFloat(tc) END AS aldpc,
  CASE WHEN tc = 0 THEN 0 ELSE toFloat(tld) / toFloat(tc) END AS aldp,
  tds,
  tw
ORDER BY rn
LIMIT 100

// UNION wordings
UNION ALL

MATCH (ba:B)
WHERE ba.rn IN [
  'A','Ai','Cl','Co','Cod','K',
  's','sr'
]
MATCH (ba)-[:H]->(lac:C)
WHERE lac.t >= '2026-01-01' AND lac.t <= '2026-02-19'
WITH collect(lac) AS cs

CALL {
  WITH cs
  UNWIND cs AS cc
  RETURN
    count(DISTINCT cc.h)             AS tc,
    coalesce(sum(cc.la), 0)          AS tla,
    coalesce(sum(cc.ld), 0)          AS tld,
    coalesce(sum(CASE WHEN cc.im = true THEN 1 ELSE 0 END), 0) AS cmt
}

CALL {
  WITH cs
  UNWIND cs AS cc
  WITH substring(cc.t,0,10) AS day, count(DISTINCT cc.h) AS cod
  ORDER BY day
  RETURN collect({day: day, count: cod}) AS cpd
}

CALL {
  WITH cs
  UNWIND cs AS cc
  WITH substring(cc.t,0,4) AS year,
       toInteger(substring(cc.t,5,2)) AS mo,
       toInteger(substring(cc.t,8,2)) AS di,
       cc
  WITH year, di +
    CASE mo
      WHEN 1 THEN 0 WHEN 2 THEN 31 WHEN 3 THEN 59 WHEN 4 THEN 90 WHEN 5 THEN 120 WHEN 6 THEN 151
      WHEN 7 THEN 181 WHEN 8 THEN 212 WHEN 9 THEN 243 WHEN 10 THEN 273 WHEN 11 THEN 304 WHEN 12 THEN 334 END
    AS doy, cc
  WITH year, toInteger((doy - 1) / 7) + 1 AS wn, cc
  WITH year + '-W' + toString(wn) AS wl, count(DISTINCT cc.h) AS ciw
  ORDER BY wl
  RETURN collect({week: wl, count: ciw}) AS cpw
}

CALL {
  WITH cs
  UNWIND cs AS cc
  WITH substring(cc.t,0,10) AS day, count(DISTINCT cc.h) AS cod
  ORDER BY cod DESC, day ASC
  LIMIT 5
  RETURN collect({day: day, count: cod}) AS tds
}

CALL {
  WITH cs
  UNWIND cs AS cc
  WITH substring(cc.t,0,4) AS year,
       toInteger(substring(cc.t,5,2)) AS mo,
       toInteger(substring(cc.t,8,2)) AS di,
       cc
  WITH year, di +
    CASE mo
      WHEN 1 THEN 0 WHEN 2 THEN 31 WHEN 3 THEN 59 WHEN 4 THEN 90 WHEN 5 THEN 120 WHEN 6 THEN 151
      WHEN 7 THEN 181 WHEN 8 THEN 212 WHEN 9 THEN 243 WHEN 10 THEN 273 WHEN 11 THEN 304 WHEN 12 THEN 334 END
    AS doy, cc
  WITH year, toInteger((doy - 1) / 7) + 1 AS wn, cc
  WITH year + '-W' + toString(wn) AS wl, count(DISTINCT cc.h) AS ciw
  ORDER BY ciw DESC, wl ASC
  LIMIT 5
  RETURN collect({week: wl, count: ciw}) AS tw
}

RETURN
  'ALL_REPOS'                           AS rn,
  tc,
  cpw                                   AS cpw,
  cpd                                   AS cpd,
  cmt,
  CASE WHEN tc = 0 THEN 0 ELSE toFloat(cmt) / toFloat(tc) END AS mrt,
  tla,
  tld,
  CASE WHEN tc = 0 THEN 0 ELSE toFloat(tla) / toFloat(tc) END AS aldpc,
  CASE WHEN tc = 0 THEN 0 ELSE toFloat(tld) / toFloat(tc) END AS aldp,
  tds,
  tw
LIMIT 1
        """
        self.graph.query(q)

        # make sure graph.query ran without raising exception
        self.env.assertTrue(True)

