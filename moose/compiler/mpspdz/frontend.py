import ast
import inspect
import textwrap
import logging
import sys
from typing import List
from dataclasses import dataclass

from compiler.mpspdz import importer
from .importer import FunctionContext


logging.basicConfig(level=logging.DEBUG)

class MpspdzFrontend:
    """Frontend for importing various entities into a Module."""
    
    def __init__(self):
        pass

    def import_global_function(self, f, ins, outs):
        print("Indices printing")
        print(ins)
        print(str(outs))
        filename = inspect.getsourcefile(f)
        source_lines, start_lineno = inspect.getsourcelines(f)
        source = "".join(source_lines)
        source = textwrap.dedent(source)
        ast_root = ast.parse(source, filename=filename)
        ast.increment_lineno(ast_root, start_lineno - 1)
        ast_fd = ast_root.body[0]
        # ast_fd.args.args

        fctx = FunctionContext(func_name="foo")
        fdimport = FunctionDefImporter(fctx, ast_fd)

        return fdimport.import_body(ins, outs)


class BaseNodeVisitor(ast.NodeVisitor):
  """Base class of a node visitor that aborts on unhandled nodes."""
  IMPORTER_TYPE = "<unknown>"

  def __init__(self, fctx):
    super().__init__()
    self.fctx = fctx

  def visit(self, node):
    return super().visit(node)

  def generic_visit(self, ast_node):
    logging.debug("UNHANDLED NODE: {}", ast.dump(ast_node))


class ExpressionImporter(BaseNodeVisitor):
  """Imports expression nodes.

  Visitor methods should either raise an exception or set self.value to the
  IR value that the expression lowers to.
  """
  IMPORTER_TYPE = "expression"
  __slots__ = [
      "value",
  ]

  def __init__(self, fctx):
    super().__init__(fctx)
    self.value = None

  def visit(self, node):
    super().visit(node)
    #assert self.value, ("ExpressionImporter did not assign a value (%r)" %
    #                    (ast.dump(node),))

  def sub_evaluate(self, sub_node):
    sub_importer = ExpressionImporter(self.fctx)
    sub_importer.visit(sub_node)
    return sub_importer.value

  def emit_constant(self, value):
    env = self.fctx.environment
    ir_const_value = env.code_py_value_as_const(value)
    if ir_const_value is NotImplemented:
      self.fctx.abort("unknown constant type '%r'" % (value,))
    self.value = ir_const_value

  def visit_Attribute(self, ast_node):
    # Import the attribute's value recursively as a partial eval if possible.
    logging.debug("visit_Attribute")

  def visit_BinOp(self, ast_node):
    logging.debug("visit_BinOp")
    left = self.sub_evaluate(ast_node.left)
    right = self.sub_evaluate(ast_node.right)
    self.value = self.fctx.get_fresh_name()
    op_string = None
    if isinstance(ast_node.op, ast.Mult):
      op_string = "mult"
    elif isinstance(ast_node.op, ast.Add):
      op_string = "add"
    assert op_string is not None
    self.fctx.emit_mlir(f"%{self.value} = !mpspdz.{op_string} %{left} %{right}: !mpspdz.sint")
  
  def visit_Call(self, ast_node):
    # Evaluate positional args.
    evaluated_args = []
    for raw_arg in ast_node.args:
      evaluated_args.append(self.sub_evaluate(raw_arg))

    # Evaluate keyword args.
    keyword_args = []
    for raw_kw_arg in ast_node.keywords:
      keyword_args.append((raw_kw_arg.arg, self.sub_evaluate(raw_kw_arg.value)))

    # Perform partial evaluation of the callee.
    logging.debug("visit_Call, please revisit: {}", ast.dump(ast_node))

  def visit_Name(self, ast_node):
    self.value = ast_node.id

  def visit_UnaryOp(self, ast_node):
    op = ast_node.op
    operand_value = self.sub_evaluate(ast_node.operand)
    logging.debug("vist_UnaryOp: {}", ast.dump(ast_node))

  if sys.version_info < (3, 8, 0):
    # <3.8 breaks these out into separate AST classes.
    def visit_Num(self, ast_node):
      self.emit_constant(ast_node.n)

    def visit_Str(self, ast_node):
      self.emit_constant(ast_node.s)

    def visit_Bytes(self, ast_node):
      self.emit_constant(ast_node.s)

    def visit_NameConstant(self, ast_node):
      self.emit_constant(ast_node.value)
  else:

    def visit_Constant(self, ast_node):
      self.emit_constant(ast_node.value)



class FunctionDefImporter(BaseNodeVisitor):
  """AST visitor for importing a function's statements.

  Handles nodes that are direct children of a FunctionDef.
  """
  IMPORTER_TYPE = "statement"
  __slots__ = [
      "ast_fd",
      "_last_was_return",
  ]
  # (TODO) fctx should be created inside the class

  def __init__(self, fctx, ast_fd):
    super().__init__(fctx)
    self.fctx = fctx
    self.ast_fd = ast_fd
    self._last_was_return = False

  def import_body(self, ins, outs):
    self.fctx.update_io(ins, outs)
    self.fctx.emit_mlir_private_inputs(self.ast_fd.args.args)
    for ast_stmt in self.ast_fd.body:
      self._last_was_return = False
      self.visit(ast_stmt)
    return self.fctx.emit_final_mlir()

  def visit_Assign(self, ast_node):
    logging.debug("visit_Assign")
    pass

  def visit_Expr(self, ast_node):
    # Evaluate the expression in the exec body.
    expr = ExpressionImporter(self.fctx)
    expr.visit(ast_node.value)

  def visit_Pass(self, ast_node):
    logging.debug("visit_Pass")
    pass

  def visit_Return(self, ast_node):
    expr = ExpressionImporter(self.fctx)
    expr.visit(ast_node.value)
    logging.debug(f"visit_Return: {expr.value}")
    self.fctx.emlit_mlir_private_output(expr.value)
    self._last_was_return = True


