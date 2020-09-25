import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import List

from moose.compiler.computation import MpspdzCallOperation
from moose.compiler.computation import MpspdzLoadOutputOperation
from moose.compiler.computation import MpspdzSaveInputOperation
from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import Placement
from moose.logger import get_logger


@dataclass
class MpspdzPlacement(Placement):
    players: List[HostPlacement]

    def __hash__(self):
        return hash(self.name)

    def compile(self, context, fn, inputs, output_placements=None, output_type=None):
        input_ops = [context.visit(expression) for expression in inputs]

        # NOTE the following two could be precomputed
        known_player_names = set(player.name for player in self.players)
        player_name_index_map = {
            player.name: i for i, player in enumerate(self.players)
        }

        # NOTE output_players and output_placement can be extracted from output_type once
        # we have placements in types
        input_player_names = [op.device_name for op in input_ops]
        [output_player_name,] = [player.name for player in output_placements]
        participating_player_names = set(input_player_names) | set([output_player_name])
        assert participating_player_names.issubset(known_player_names)

        mlir_string = compile_to_mlir(
            fn,
            input_indices=[
                player_name_index_map[player_name] for player_name in input_player_names
            ],
            output_index=player_name_index_map[output_player_name],
        )
        get_logger().debug(mlir_string)

        # generate one save operation for each input player
        save_input_ops = [
            MpspdzSaveInputOperation(
                device_name=player_name,
                name=context.get_fresh_name("mpspdz_save_input_op"),
                inputs={
                    f"arg{i}": matching_input_op.output
                    for i, matching_input_op in enumerate(
                        input_op
                        for input_op in input_ops
                        if input_op.device_name == player_name
                    )
                },
                output=context.get_fresh_name("mpspdz_save_input"),
                player_index=player_name_index_map[player_name],
            )
            for player_name in set(input_player_names)
        ]

        # generate operations for all participating players to invoke MP-SPDZ
        # TODO we probably need to have a control dependency of some sort here
        # operation for the player to execute MP-SPDZ; I suggest we take the Chain
        # approach used in TFRT (which is similar to units)
        call_ops = [
            MpspdzCallOperation(
                device_name=player_name,
                name=context.get_fresh_name("mpspdz_call_op"),
                inputs={
                    f"call_control{i}": context.maybe_add_networking(
                        save_op, player_name
                    ).output
                    for i, save_op in enumerate(save_input_ops)
                },
                output=context.get_fresh_name("mpspdz_call"),
                player_index=player_name_index_map[player_name],
                mlir=mlir_string,
                bytecode=None,  # TODO
            )
            for player_name in participating_player_names
        ]

        # operation for loading the output
        # TODO also need control dependency here
        load_output_op = MpspdzLoadOutputOperation(
            device_name=output_player_name,
            name=context.get_fresh_name("mpspdz_load_output_op"),
            inputs={
                f"load_control{i}": context.maybe_add_networking(
                    call_op, output_player_name
                ).output
                for i, call_op in enumerate(call_ops)
                if call_op.device_name == output_player_name
            },
            output=context.get_fresh_name("mpspdz_output"),
            player_index=player_name_index_map[output_player_name],
        )

        context.operations += save_input_ops + call_ops
        return load_output_op


def compile_to_mlir(fn, input_indices, output_index):
    fn_ast = extract_ast(fn)
    input_names = [arg.arg for arg in fn_ast.args.args]

    module = MlirModule()

    # add main function
    main_function = MlirFunction(name="main", args=[], type=None)
    get_input_ops = [
        MlirOperation(
            name=arg_name, value=f"mpspdz.get_input_from {index}", type="!mpspdz.sint",
        )
        for arg_name, index in zip(input_names, input_indices)
    ]
    call_op = MlirOperation(
        name=main_function.get_fresh_name(),
        value=f"mpspdz.call @{fn_ast.name} %x, %y, %z",
        type="!mpspdz.sint",
    )
    reveal_op = MlirOperation(
        name=None,
        value=f"mpspdz.reveal_to %{call_op.name} {output_index}",
        type="!mpspdz.sint -> !mpspdz.cint",
    )
    main_function.add_operations(*get_input_ops, call_op, reveal_op)
    module.add_function(main_function)

    # add function for `fn`
    function_visitor = FunctionVisitor(module)
    function_visitor.visit(fn_ast)

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
        return f"{self.name} {type_annotation}"


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
                type=None,
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
                    value=f"mpspdz.mul %{left} %{right}",
                    type="!mpspdz.sint",
                )
            )
        elif isinstance(ast_node.op, ast.Add):
            self.function.add_operation(
                MlirOperation(
                    name=self.value,
                    value=f"mpspdz.add %{left} %{right}",
                    type="!mpspdz.sint",
                )
            )
        else:
            raise NotImplementedError()

    def visit_Name(self, ast_node):
        self.value = ast_node.id
