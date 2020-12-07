import ast
import inspect
import textwrap

from moose.computation.mpspdz import MpspdzCallOperation
from moose.computation.mpspdz import MpspdzLoadOutputOperation
from moose.computation.mpspdz import MpspdzPlacement
from moose.computation.mpspdz import MpspdzSaveInputOperation
from moose.computation.standard import ApplyFunctionOperation
from moose.logger import get_logger


class MpspdzApplyFunctionPass:
    def run(self, computation, context):
        ops_to_replace = []
        for op in computation.operations.values():
            if not isinstance(op, ApplyFunctionOperation):
                continue
            placement = computation.placement(op.placement_name)
            if not isinstance(placement, MpspdzPlacement):
                continue
            ops_to_replace += [op]

        performed_changes = False
        for op in ops_to_replace:
            self.lower(
                op=op, computation=computation, context=context,
            )
            performed_changes = True

        return computation, performed_changes

    def lower(self, op, computation, context):
        mpspdz_placement = computation.placement(op.placement_name)
        input_ops = computation.find_sources(op)
        output_ops = computation.find_destinations(op)

        input_placement_names = [op.placement_name for op in input_ops]
        output_placement_names = [op.placement_name for op in output_ops]
        compute_placement_names = mpspdz_placement.player_names
        assert set(compute_placement_names) == set(
            input_placement_names + output_placement_names
        )

        assert len(output_ops) == 1  # required by MP-SPDZ
        output_op = output_ops[0]
        assert len(output_placement_names) == 1  # required by MP-SPDZ
        output_placement_name = output_placement_names[0]

        index_map = {
            placement_name: i
            for i, placement_name in enumerate(compute_placement_names)
        }
        coordinator = compute_placement_names[0]  # required by MP-SPDZ

        mlir_string = compile_to_mlir(
            op.fn,
            input_indices=[
                index_map[placement_name] for placement_name in input_placement_names
            ],
            output_index=index_map[output_placement_name],
        )
        invocation_key = context.get_fresh_name("invocation_key")

        # generate one save operation for each input player
        save_input_ops = [
            MpspdzSaveInputOperation(
                placement_name=placement_name,
                name=context.get_fresh_name("mpspdz_save_input"),
                inputs={
                    f"arg{i}": matching_input_op.name
                    for i, matching_input_op in enumerate(
                        input_op
                        for input_op in input_ops
                        if input_op.placement_name == placement_name
                    )
                },
                player_index=index_map[placement_name],
                invocation_key=invocation_key,
            )
            for placement_name in set(input_placement_names)
        ]
        computation.add_operations(save_input_ops)

        # generate operations for all participating players to invoke MP-SPDZ
        call_ops = [
            MpspdzCallOperation(
                placement_name=placement,
                name=context.get_fresh_name("mpspdz_call"),
                inputs={
                    f"call_control{i}": save_op.name
                    for i, save_op in enumerate(save_input_ops)
                },
                player_index=index_map[placement],
                num_players=len(compute_placement_names),
                mlir=mlir_string,
                invocation_key=invocation_key,
                coordinator=coordinator,
                protocol="mascot",
            )
            for placement in compute_placement_names
        ]
        computation.add_operations(call_ops)

        # operation for loading the output
        load_output_op = MpspdzLoadOutputOperation(
            placement_name=output_placement_name,
            name=context.get_fresh_name("mpspdz_load_output"),
            inputs={
                f"load_control{i}": call_op.name for i, call_op in enumerate(call_ops)
            },
            player_index=index_map[output_placement_name],
            invocation_key=invocation_key,
        )
        computation.add_operation(load_output_op)

        # rewire output
        for arg_name in output_op.inputs.keys():
            op_name = output_op.inputs[arg_name]
            if op_name == op.name:
                output_op.inputs[arg_name] = load_output_op.name
        computation.remove_operation(op.name)


def compile_to_mlir(fn, input_indices, output_index):
    fn_ast = extract_ast(fn)
    input_names = [arg.arg for arg in fn_ast.args.args]

    module = MlirModule()

    # add function for `fn`
    function_visitor = FunctionVisitor(module)
    function_visitor.visit(fn_ast)
    fn_mlir = function_visitor.function

    # add main function
    main_function = MlirFunction(name="main", args=[], type=None)
    get_input_ops = [
        MlirOperation(
            name=arg_name, value=f"mpspdz.get_input_from {index}", type="!mpspdz.sint"
        )
        for arg_name, index in zip(input_names, input_indices)
    ]
    arg_names = ", ".join(f"%{arg.name}" for arg in fn_mlir.args)
    arg_types = ", ".join(arg.type for arg in fn_mlir.args)
    call_op = MlirOperation(
        name=main_function.get_fresh_name(),
        value=f"mpspdz.call @{fn_mlir.name}({arg_names})",
        type=f"({arg_types}) -> {fn_mlir.type}",
    )
    reveal_op = MlirOperation(
        name=None,
        value=f"mpspdz.reveal_to %{call_op.name} {output_index}",
        type="!mpspdz.sint",
    )
    return_op = MlirOperation(name=None, value="mpspdz.return", type=None)
    main_function.add_operations(*get_input_ops, call_op, reveal_op, return_op)
    module.add_function(main_function)

    # emit and return MLIR code
    return module.emit_mlir()


def extract_ast(fn):
    filename = inspect.getsourcefile(fn)
    source_lines, start_lineno = inspect.getsourcelines(fn)
    source = "".join(source_lines)
    source = textwrap.dedent(source)
    ast_root = ast.parse(source, filename=filename)
    ast.increment_lineno(ast_root, start_lineno - 1)
    ast_fn = ast_root.body[0]
    return ast_fn


class MlirModule:
    def __init__(self, name=None):
        self.name = name
        self.functions = []
        self.tmp_counter = 0

    def add_function(self, function):
        self.functions.append(function)

    def emit_mlir(self, indent_spaces=0):
        indent = " " * indent_spaces
        name = f" @{self.name}" if self.name else ""
        return (
            f"{indent}module{name} {{\n"
            + "\n".join(
                function.emit_mlir(indent_spaces + 2) for function in self.functions
            )
            + f"\n{indent}}}"
        )


class MlirFunction:
    def __init__(self, name, args, type):
        self.name = name
        self.args = args
        self.type = type
        self.ops = []
        self.tmp_counter = 0

    def names(self):
        return [op.name for op in self.ops] + [arg.name for arg in self.args]

    def get_fresh_name(self):
        while f"tmp{self.tmp_counter}" in self.names():
            self.tmp_counter += 1
        return f"tmp{self.tmp_counter}"

    def add_operation(self, operation):
        self.ops.append(operation)

    def add_operations(self, *operations):
        self.ops.extend(operations)

    def emit_mlir(self, indent_spaces):
        indent = " " * indent_spaces
        args = ", ".join(arg.emit_mlir() for arg in self.args)
        return_type = f" -> {self.type}" if self.type else ""
        ops = "\n".join(op.emit_mlir(indent_spaces + 2) for op in self.ops)
        return (
            f"{indent}mpspdz.func @{self.name}({args}){return_type} {{\n"
            + f"{ops}\n"
            + f"{indent}}}"
        )


class MlirOperation:
    def __init__(self, name, value, type):
        self.name = name
        self.value = value
        self.type = type

    def emit_mlir(self, indent_spaces):
        indent = " " * indent_spaces
        prefix = f"%{self.name} = " if self.name else ""
        postfix = f" : {self.type}" if self.type else ""
        return f"{indent}{prefix}{self.value}{postfix}"


class MlirArg:
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def emit_mlir(self):
        type_annotation = f": {self.type}" if self.type else ""
        return f"%{self.name} {type_annotation}"


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, module):
        self.module = module
        self.function = None

    def generic_visit(self, ast_node):
        get_logger().error(f"Unhandled: {ast.dump(ast_node)}")

    def visit_FunctionDef(self, ast_node):
        arg_names = [arg.arg for arg in ast_node.args.args]
        self.function = MlirFunction(
            name=ast_node.name,
            args=[
                MlirArg(name=arg_name, type="!mpspdz.sint") for arg_name in arg_names
            ],
            type="!mpspdz.sint",
        )
        for ast_stmt in ast_node.body:
            self.visit(ast_stmt)
        self.module.add_function(self.function)

    def visit_Expr(self, ast_node):
        expression_visitor = ExpressionVisitor(self.function)
        expression_visitor.visit(ast_node.value)

    def visit_Return(self, ast_node):
        expression_visitor = ExpressionVisitor(self.function)
        expression_visitor.visit(ast_node.value)
        self.function.add_operation(
            MlirOperation(
                name=None,
                value=f"mpspdz.return %{expression_visitor.value}",
                type="!mpspdz.sint",
            )
        )


class ExpressionVisitor(ast.NodeVisitor):
    def __init__(self, function):
        self.function = function
        self.value = None

    def sub_visit(self, sub_node):
        sub_importer = ExpressionVisitor(self.function)
        sub_importer.visit(sub_node)
        return sub_importer.value

    def visit_BinOp(self, ast_node):
        left = self.sub_visit(ast_node.left)
        right = self.sub_visit(ast_node.right)
        self.value = self.function.get_fresh_name()
        if isinstance(ast_node.op, ast.Mult):
            self.function.add_operation(
                MlirOperation(
                    name=self.value,
                    value=f"mpspdz.mul %{left}, %{right}",
                    type="!mpspdz.sint",
                )
            )
        elif isinstance(ast_node.op, ast.Add):
            self.function.add_operation(
                MlirOperation(
                    name=self.value,
                    value=f"mpspdz.add %{left}, %{right}",
                    type="!mpspdz.sint",
                )
            )
        else:
            raise NotImplementedError()

    def visit_Name(self, ast_node):
        self.value = ast_node.id
