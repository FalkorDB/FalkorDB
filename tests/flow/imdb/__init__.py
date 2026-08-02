class QueryInfo(object):
    """
    This class contains the needed data about a query
    """

    def __init__(self, query=None, description=None, expected_result=None, reversible=True):
        """
        QueryInfo contructor

        :param query: The query itself (string)
        :param description: The information about what the query does (string)
        :param expected_result: The expected result of the query (list of lists, where the first list
                                is the columns names, and the rest is the result)
        """

        self.query = query
        self.description = description
        self.expected_result = expected_result
        self.reversible = reversible

