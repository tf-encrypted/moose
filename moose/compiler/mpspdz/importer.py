#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Importers for populating MLIR from AST.
"""
import ast
import sys
import traceback
import logging

__all__ = [
    "FunctionContext",
]

class FunctionContext:
  """Accounting information for importing a function."""

  def __init__(self, func_name): # pass args
    self.variable_counter = 0
    self.mlir_string = ""
    self.func_name = func_name

  def update_io(self, in_ids, out_ids):
    self.in_ids = in_ids
    self.out_ids = out_ids

  def get_fresh_name(self) -> str:
    self.variable_counter += 1
    return "v" + str(self.variable_counter)

  def emit_mlir(self, instruction):
    self.mlir_string += instruction + "\n"
    logging.debug(f"\n{self.mlir_string}")
  
  def emit_final_mlir(self):
    return "module {\n" + f"mpspdz.func @{self.func_name}()" + " {\n" + f"{self.mlir_string}" + "\n}" + "}"
  
  def emit_mlir_private_inputs(self, var_names):
    party_ids = self.in_ids
    assert(len(party_ids) == len(var_names))
    for i in range(len(party_ids)):
      self.mlir_string += f"%{var_names[i].arg} = mpspdz.get_input_from %{party_ids[i]}: !mpspdz.sint" + "\n"
  
  def emlit_mlir_private_output(self, var_name):
    for party_id in self.out_ids:
      self.mlir_string += f"!mpspdz.reveal_to %{var_name} {party_id}: !mpspdz.unit \n"
    

