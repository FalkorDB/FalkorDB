from falkordb import FalkorDB
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Tree, ProgressBar, Label, Input
from textual.widgets.tree import TreeNode
from textual.reactive import reactive

db = FalkorDB()


class ReactiveLabel(Label):
    text_value = reactive("Initial text")

    def watch_text_value(self, new_value: str) -> None:
        self.update(new_value)


class QueryVisualizerApp(App):
    CSS = "Label { margin: 0 2 0 0; }" "Horizontal { height: auto; }"
    current_index = reactive(0)
    last_query = None

    def __init__(self, query: str):
        super().__init__()
        self.record = []
        self.query_string = []
        self.run_query("g", query)

    def compose(self) -> ComposeResult:
        query = ReactiveLabel(id="query_label")
        query.text_value = self.query_string[
            self.last_query if self.last_query is not None else -1
        ]
        yield query
        yield Input(placeholder="Search", compact=True, id="search_input")
        self.tree_map: dict[str, (TreeNode, str)] = {}
        tree = Tree(self.record[1][0][2])
        tree.root.allow_expand = False
        self.tree_map[self.record[1][0][0]] = (tree.root, self.record[1][0])
        for row in self.record[1][1:]:
            self.tree_map[row[0]] = (self.tree_map[row[1]][0].add_leaf(row[2]), row)
        if len(self.record[0]) > 1:
            row = self.record[0][0]
            if row[1] == 0:
                self.tree_map[row[0]][0].label += f" | Error: {row[2]}"
            elif row[1] == 1:
                env = {a[0]: a[1] for a in zip(self.tree_map[row[0]][1][3], row[2])}
                self.tree_map[row[0]][0].label += f" | Env: {env}"
        tree.root.expand_all()
        yield tree
        label = ReactiveLabel(id="step_label")
        label.text_value = f"Step: {self.current_index + 1}/{len(self.record[0])}"
        yield Horizontal(
            label,
            ProgressBar(len(self.record[0]) - 1, show_eta=0, show_percentage=False),
        )
        graph_name = Input(placeholder="Enter graph", id="graph_name", max_length=10)
        graph_name.styles.width = 20
        yield Horizontal(
            graph_name,
            Input(placeholder="Enter query", id="input_query"),
        )

    def on_ready(self):
        row = self.record[0][0]
        self.query_one(Tree).select_node(self.tree_map[row[0]][0])

    def run_query(self, graph_name: str, query: str) -> None:
        try:
            record = db.execute_command("GRAPH.RECORD", graph_name, query)
            self.record = record
            self.current_index = 0
            self.query_string.append(query)
        except:
            pass

        self.refresh(recompose=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.control.id == "search_input":
            for i, row in enumerate(self.record[0]):
                if i <= self.current_index:
                    continue
                env = {a[0]: a[1] for a in zip(self.tree_map[row[0]][1][3], row[2])}
                try:
                    if eval(event.value, env):
                        self.current_index = i
                        self.query_one(ProgressBar).progress = i
                        self.update_tree()
                        break
                except:
                    pass
        elif event.control.id == "input_query":
            self.last_query = None
            graph = self.query_one("#graph_name").value
            self.run_query(graph, event.value)

    def update_tree(self):
        self.get_widget_by_id("step_label").text_value = (
            f"Step: {self.current_index + 1}/{len(self.record[0])}"
        )
        for node in self.tree_map.values():
            node[0].label = node[1][2]
        row = self.record[0][self.current_index]
        if row[1] == 0:
            self.tree_map[row[0]][0].label += f" | Error: {row[2]}"
        elif row[1] == 1:
            env = {a[0]: a[1] for a in zip(self.tree_map[row[0]][1][3], row[2])}
            self.tree_map[row[0]][0].label += f" | Env: {env}"
        self.query_one(Tree).select_node(self.tree_map[row[0]][0])

    def on_key(self, event) -> None:
        if self.query_one(Tree).has_focus:
            if event.key == "left":
                if self.current_index > 0:
                    self.current_index = self.current_index - 1
                    self.query_one(ProgressBar).advance(-1)
                    self.update_tree()
            elif event.key == "right":
                if self.current_index < len(self.record[0]) - 1:
                    self.current_index = self.current_index + 1
                    self.query_one(ProgressBar).advance(1)
                    self.update_tree()
        elif self.get_widget_by_id("input_query").has_focus:
            if event.key == "up":
                if self.last_query is None:
                    self.last_query = len(self.query_string) - 1
                else:
                    self.last_query = max(0, self.last_query - 1)
                self.get_widget_by_id("input_query").value = self.query_string[
                    self.last_query
                ]
            elif event.key == "down":
                if self.last_query is not None:
                    if self.last_query == len(self.query_string) - 1:
                        self.last_query = None
                        self.get_widget_by_id("input_query").clear()
                    else:
                        self.last_query = min(
                            len(self.query_string) - 1, self.last_query + 1
                        )
                        self.get_widget_by_id("input_query").value = (
                            self.query_string[self.last_query]
                            if self.last_query is not None
                            else ""
                        )


if __name__ == "__main__":
    app = QueryVisualizerApp(
        "UNWIND range(1, 10) AS x UNWIND range(x, 10) AS y RETURN x, y"
    )
    app.run()
