import os
import warnings

import pytest

# The default warnings.warn prints itself, this function should override the default behavior
# Output example: "UserWarning: Maximum runtime for query "My friends?" was: 0.378, but should be 0.2"
def _warning_formater(message, category, filename, lineno, file=None, line=None):
    return '%s: %s\n' % (category.__name__, message)

warnings.formatwarning = _warning_formater

class FlowTestsBase(object):

    def _assert_only_expected_results_are_in_actual_results(self,
                                                           actual_result,
                                                           query_info):
        actual_result_set = []
        if actual_result.result_set is not None:
            actual_result_set = actual_result.result_set

        # Assert number of results.
        assert len(actual_result_set) == len(query_info.expected_result)

        # Assert actual values vs expected values.
        for res in query_info.expected_result:
            assert res in actual_result_set

    def _assert_actual_results_contained_in_expected_results(self,
                                                             actual_result,
                                                             query_info,
                                                             num_contained_results):
        actual_result_set = actual_result.result_set

        # Assert num results.
        assert len(actual_result_set) == num_contained_results

        # Assert actual values vs expected values.
        expected_result = query_info.expected_result
        count = len([res for res in expected_result if res in actual_result_set])

        # Assert number of different results is as expected.
        assert count == num_contained_results


    def _assert_resultset_and_expected_mutually_included(self, actual_result, query_info):
        actual_result_set = []
        if actual_result.result_set is not None:
            actual_result_set = actual_result.result_set

        # Assert number of results.
        assert len(actual_result_set) == len(query_info.expected_result)

        # Assert actual values vs expected values.
        for res in query_info.expected_result:
            assert res in actual_result_set
        
        # Assert expected values vs actual values.
        for res in actual_result_set:
            assert res in query_info.expected_result

    def _assert_resultset_equals_expected(self, actual_result, query_info):
        actual_result_set = actual_result.result_set or []
        assert actual_result_set == query_info.expected_result

    # function which run the query and expects an specific error message
    def _assert_exception(self, graph, query, expected_err_msg):
        with pytest.raises(Exception) as excinfo:
            graph.query(query)
        assert expected_err_msg in str(excinfo.value)
