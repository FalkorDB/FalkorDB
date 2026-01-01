class ReversePattern(object):
    def __init__(self):
        pass

    def consume_node(self, q):
        if q[0] != "(":
            return None
        
        end = q.find(")")
        if end == -1:
            return None

        return q[0:end+1]

    def consume_edge(self, q):
        if q[0] != "[":
            return None

        end = q.find("]")
        if end == -1:
            return None

        return q[0:end+1]

    def consume_ltr_arrow(self, q):
        if q[:2] != "->":
            return None
        # Reversed on purpose!
        return "<-"

    def consume_rtl_arrow(self, q):
        if q[:2] != "<-":
            return None
        # Reversed on purpose!
        return "->"

    def consume_dash(self, q):
        if q[0] != "-":
            return None
        return q[0:1]

    def consume_space(self, q):
        if q[0] != " ":
            return None
        return q[0:1]

    # TODO update to handle WITH clauses
    def reverse_query_pattern(self, q):
        q = q.replace('\r', '')
        q = q.replace('\n', '')
        q = q.replace('\t', '')
        q = ' '.join(q.split())

        start = q.find("MATCH")
        if start == -1:
            print("MATCH not found")
            return None

        reversed_query = ""

        # Find the leftmost node.
        start = q.find("(")

        stop_words = ["WHERE", "RETURN", "SET", "DELETE"]
        for stop in stop_words:
            end = q.find(stop)
            if end != -1:
                break

        if end == -1:
            print("stop word was not found")
            return None
        end -= 1

        # Order of consumers is important!
        consumers = [self.consume_space,
                     self.consume_node,
                     self.consume_edge,
                     self.consume_ltr_arrow,
                     self.consume_rtl_arrow,
                     self.consume_dash]
        
        # Reverse entities
        while (start <= end):
            entity = None
            for c in consumers:
                # Call current consumer.
                entity = c(q[start:])
                if entity:
                    # Manged to progress, append current consumed entity.
                    start += len(entity)
                    reversed_query = entity + reversed_query
                    break
            
            # Did not managed to progress.
            if entity is None:
                break

        reversed_query = "MATCH" + reversed_query + q[end:]
        return reversed_query
