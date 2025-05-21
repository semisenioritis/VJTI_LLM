

# Final ready code

# !pip install graphviz
# !git clone https://github.com/yt-dlp/yt-dlp

import ast
import os
from graphviz import Digraph
import html



class FunctionInfo:
    def __init__(self, name, lineno, parent_function_path=None, parent_class_path=None, file_path=None):
        self.name = name
        self.lineno = lineno
        self.args = []
        self.docstring = None
        self.decorators = []
        self.is_async = False
        self.body = None
        self.parent_function_path = parent_function_path or []
        self.parent_class_path = parent_class_path or []
        self.file_path = file_path

    def __repr__(self):
        return f"<FunctionInfo name={self.name} lineno={self.lineno} func_path={self.parent_function_path} class_path={self.parent_class_path} file={self.file_path}>"


    def qualified_id(self):
        return tuple(self.parent_class_path + self.parent_function_path + [self.name])

def extract_functions(source_code, file_path=None):
    tree = ast.parse(source_code)
    functions = []
    class_defs = {}

    mod = FunctionInfo(name="__root__", lineno=1,
                      parent_function_path=[],
                      parent_class_path=[], file_path=file_path)
    mod.body = tree.body
    functions.append(mod)


    class ClassCollector(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            class_defs[node.name] = node
            self.generic_visit(node)

    ClassCollector().visit(tree)


    def resolve_class_ancestry(class_node, seen=None):
        seen = seen or set()
        ancestry = []
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id not in seen:
                seen.add(base.id)
                ancestry.append(base.id)
                base_node = class_defs.get(base.id)
                if base_node:
                    ancestry.extend(resolve_class_ancestry(base_node, seen))
        return ancestry

    def visit(node, function_stack, class_stack):
        if isinstance(node, ast.ClassDef):
            class_stack.append(node.name)
            for child in node.body:
                visit(child, function_stack, class_stack)
            class_stack.pop()

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current_class_path = []
            if class_stack:
                cls = class_defs.get(class_stack[-1])
                if cls:
                    current_class_path = [class_stack[-1]] + resolve_class_ancestry(cls)

            fn_info = FunctionInfo(
                name=node.name,
                lineno=node.lineno,
                parent_function_path=list(function_stack),
                parent_class_path=current_class_path,
                file_path=file_path
            )
            fn_info.is_async = isinstance(node, ast.AsyncFunctionDef)
            fn_info.body = node.body
            functions.append(fn_info)

            function_stack.append(node.name)
            for child in ast.iter_child_nodes(node):
                visit(child, function_stack, class_stack)
            function_stack.pop()
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, function_stack, class_stack)

    visit(tree, [], [])
    return functions


def collect_called_functions(body):
    called = set()
    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            self.generic_visit(node)
    for stmt in body:
        CallVisitor().visit(stmt)
    return called

def build_adjacency_matrix(functions):
    id_to_index = {fn.qualified_id(): i for i, fn in enumerate(functions)}
    n = len(functions)
    matrix = [[0]*n for _ in range(n)]

    for i, fn in enumerate(functions):
        if not fn.body: continue
        called_names = collect_called_functions(fn.body)

        for j, other_fn in enumerate(functions):
            if other_fn.name in called_names:
                if fn.name == "__root__" and other_fn.parent_function_path == [] and fn.file_path == other_fn.file_path:
                    matrix[i][j] = 1

                elif fn.parent_function_path + [fn.name] == other_fn.parent_function_path[:len(fn.parent_function_path)+1]:
                    matrix[i][j] = 1


    return matrix

def extract_imports(source_code):
    tree = ast.parse(source_code)
    imports = {}
    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = (alias.name, None)
        def visit_ImportFrom(self, node):
            mod = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = (mod, alias.name)
    ImportVisitor().visit(tree)
    return imports

def add_interfile_edges_all(matrix, functions, root_dir):
    # 1) build imports map for every file
    imports_map = {}
    for dp, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.endswith(".py"): continue
            abs_p = os.path.join(dp, fn)
            rel_p = os.path.relpath(abs_p, root_dir)
            try:
                src = open(abs_p, encoding="utf-8").read()
                imports_map[rel_p] = extract_imports(src)
            except:
                imports_map[rel_p] = {}

    # 2) build lookup for (module, name) -> indices
    lookup = {}
    for idx, fn in enumerate(functions):
        if fn.file_path:
            mod = fn.file_path[:-3].replace(os.sep, ".")
            lookup.setdefault((mod, fn.name), []).append(idx)

    # 3) scan each function’s own body
    for i, fn in enumerate(functions):
        if not fn.file_path or fn.name=="__root__": continue
        imps = imports_map.get(fn.file_path, {})
        class CallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.called = set()
            def visit_Call(self, node):
                self._maybe(node.func)
            def _maybe(self, func):
                if isinstance(func, ast.Name) and func.id in imps:
                    mod, orig = imps[func.id]
                    name = orig or func.id
                    self.called.add((mod, name))
                elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    base = func.value.id
                    if base in imps and imps[base][1] is None:
                        self.called.add((imps[base][0], func.attr))
            def visit_FunctionDef(self, node): pass
            def visit_AsyncFunctionDef(self, node): pass
            def visit_ClassDef(self, node): pass

        v = CallVisitor()
        for stmt in fn.body or []:
            v.visit(stmt)

        for key in v.called:
            for j in lookup.get(key, []):
                matrix[i][j] = 1

    return matrix


import os

def collect_all_functions_from_dir(root_dir):
    all_functions = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".py"):
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root_dir)
                try:
                    with open(abs_path, encoding="utf-8") as f:
                        code = f.read()
                    funcs = extract_functions(code, file_path=rel_path)
                    all_functions.extend(funcs)
                except Exception as e:
                    print(f"Failed to parse {rel_path}: {e}")
    return all_functions


# proj_file_root = "/content/demo_project"
# 
# all_funcs = collect_all_functions_from_dir(proj_file_root)
# adj = build_adjacency_matrix(all_funcs)
# 
# matrix = add_interfile_edges_all(adj, all_funcs, root_dir=proj_file_root)

# for row in matrix:
#     print(row)





def visualize_function_call_graph(functions, matrix, output_file="function_call_graph"):
    dot = Digraph(format="png")

    # Assign unique node ids
    node_ids = {}
    for i, fn in enumerate(functions):
        label = ".".join(fn.qualified_id())
        node_id = f"n{i}"
        node_ids[i] = node_id
        dot.node(node_id, label)

    # Add edges for function calls
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val:
                dot.edge(node_ids[i], node_ids[j])

    dot.render(output_file, view=True)

# visualize_function_call_graph(all_funcs, matrix)



def escape(s):
    return html.escape(str(s))

def visualize_function_call_graph(functions, matrix, output_file="function_call_graph"):
    dot = Digraph(format="png")
    dot.attr('node', shape='plaintext')

    node_ids = {}
    for i, fn in enumerate(functions):
        node_id = f"n{i}"
        node_ids[i] = node_id

        label = f"""<<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0">
            <TR><TD COLSPAN="2"><B>{escape('.'.join(fn.qualified_id()))}</B></TD></TR>
            <TR><TD>Line</TD><TD>{fn.lineno}</TD></TR>
            <TR><TD>File</TD><TD>{escape(fn.file_path or 'N/A')}</TD></TR>
            <TR><TD>Class</TD><TD>{escape(fn.parent_class_path or 'N/A')}</TD></TR>
            <TR><TD>Function</TD><TD>{escape(fn.parent_function_path or 'N/A')}</TD></TR>
        </TABLE>>"""

        dot.node(node_id, label)

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val:
                dot.edge(node_ids[i], node_ids[j])

    dot.render(output_file, view=True)

# visualize_function_call_graph(all_funcs, matrix)

"""# Demo



"""



proj_file_root = "/content/yt-dlp"

all_funcs = collect_all_functions_from_dir(proj_file_root)
adj = build_adjacency_matrix(all_funcs)

matrix = add_interfile_edges_all(adj, all_funcs, root_dir=proj_file_root)

for row in matrix:
    print(row)



def visualize_function_call_graph(functions, matrix, output_file="function_call_graph"):
    dot = Digraph(format="png")

    # Assign unique node ids
    node_ids = {}
    for i, fn in enumerate(functions):
        label = ".".join(fn.qualified_id())
        node_id = f"n{i}"
        node_ids[i] = node_id
        dot.node(node_id, label)

    # Add edges for function calls
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val:
                dot.edge(node_ids[i], node_ids[j])

    dot.render(output_file, view=True)

visualize_function_call_graph(all_funcs, matrix)



def escape(s):
    return html.escape(str(s))

def visualize_function_call_graph(functions, matrix, output_file="function_call_graph"):
    dot = Digraph(format="png")
    dot.attr('node', shape='plaintext')

    node_ids = {}
    for i, fn in enumerate(functions):
        node_id = f"n{i}"
        node_ids[i] = node_id

        label = f"""<<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0">
            <TR><TD COLSPAN="2"><B>{escape('.'.join(fn.qualified_id()))}</B></TD></TR>
            <TR><TD>Line</TD><TD>{fn.lineno}</TD></TR>
            <TR><TD>File</TD><TD>{escape(fn.file_path or 'N/A')}</TD></TR>
            <TR><TD>Class</TD><TD>{escape(fn.parent_class_path or 'N/A')}</TD></TR>
            <TR><TD>Function</TD><TD>{escape(fn.parent_function_path or 'N/A')}</TD></TR>
        </TABLE>>"""

        dot.node(node_id, label)

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val:
                dot.edge(node_ids[i], node_ids[j])

    dot.render(output_file, view=True)

visualize_function_call_graph(all_funcs, matrix)


