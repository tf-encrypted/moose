use crate::computation::*;

/* Desired mutation definition

// To replace one-to-one
fn replace_add_with_mul(...)
  magic {
      x + y => x * y,
      other => other
  }

// To replace one-to-many
fn replace_add_with_double_add(...)
  magic {
      x + y => [
          z = x + y,
          x + y
      ],
      other => other
  }

// To replace many-to-many ???

*/

/// Common definitions
trait OperandContext {}

trait OneToOneGraphMutator<'a, C: 'a + OperandContext> {
    fn binary(op: &'a Operation) -> (Operator, Option<C>, Option<C>);
}

struct SymbolicOperandContext<'a> {
    operation: &'a Operation,
    ty: Ty,
}

/// Symbolic mode - replacing nodes within the graph
struct SymbolicOneToOneGraphMutator {}

impl<'a> OperandContext for SymbolicOperandContext<'a> {}

impl<'a> OneToOneGraphMutator<'a, SymbolicOperandContext<'a>> for SymbolicOneToOneGraphMutator {
    fn binary(
        op: &'a Operation,
    ) -> (
        Operator,
        Option<SymbolicOperandContext>,
        Option<SymbolicOperandContext>,
    ) {
        match op {
            Operation {
                kind: Operator::StdAdd(a),
                ..
            } => (
                op.kind.clone(),
                Some(SymbolicOperandContext {
                    operation: op,
                    ty: a.lhs,
                }),
                Some(SymbolicOperandContext {
                    operation: op,
                    ty: a.rhs,
                }),
            ),
            _ => (op.kind.clone(), None, None),
        }
    }
}

/// Here we define that in the Symbolic Contex multiplication should yield an Operation node for the graph.
impl std::ops::Mul<SymbolicOperandContext<'_>> for SymbolicOperandContext<'_> {
    type Output = Operation;
    fn mul(self, rhs: SymbolicOperandContext) -> Operation {
        Operation {
            name: self.operation.name.clone(),
            kind: Operator::StdMul(StdMulOp {
                lhs: self.ty,
                rhs: rhs.ty,
            }),
            inputs: self.operation.inputs.clone(),
            placement: self.operation.placement.clone(),
        }
    }
}

/// Concrete mode starts
struct ConcreteOneToOneGraphMutator {}

struct ConcreteOperandContext {}

impl OperandContext for ConcreteOperandContext {}

/// This is almost exactly the same as the impl for the Symbolic context. Perhaps there is room to write a simple macross
impl<'a> OneToOneGraphMutator<'a, ConcreteOperandContext> for ConcreteOneToOneGraphMutator {
    fn binary(
        op: &Operation,
    ) -> (
        Operator,
        Option<ConcreteOperandContext>,
        Option<ConcreteOperandContext>,
    ) {
        match op {
            Operation {
                kind: Operator::StdAdd(_),
                ..
            } => (
                op.kind.clone(),
                Some(ConcreteOperandContext {}),
                Some(ConcreteOperandContext {}),
            ),
            _ => (op.kind.clone(), None, None),
        }
    }
}

/// Here we define that multiplication in the concrete context is just a multiplication
impl std::ops::Mul<ConcreteOperandContext> for ConcreteOperandContext {
    type Output = fn(Vec<Value>) -> Value;
    fn mul(self, _rhs: ConcreteOperandContext) -> fn(Vec<Value>) -> Value {
        |args| args[0].clone() * args[1].clone()
    }
}

/// Sugar to multiple `Value`. If we like it, it probably belongs to computation.rs
impl std::ops::Mul<Value> for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Float32Tensor(lhs), Value::Float32Tensor(rhs)) => {
                Value::Float32Tensor(lhs * rhs)
            }
            _ => panic!("Unexpected value type"),
        }
    }
}

/// Finally the definition of the mutation step. The whole reason for all the code above
///
/// TODO: Simplify method signature. It really only needs one type parameter, not two.
fn replace_add_with_mul<
    'a,
    C: 'a + OperandContext + std::ops::Mul<C>,
    M: OneToOneGraphMutator<'a, C>,
>(
    op: &'a Operation,
) -> Option<C::Output> {
    match M::binary(op) {
        // Here mutator asks to replace binary operation StdAdd with `x * y`. This being one-liner was the goal.
        (Operator::StdAdd(_), Some(x), Some(y)) => Some(x * y),
        // None means the mutator made no changes
        _ => None,
    }
}

/// The most primitive form of graph traversal - just visit the nodes in order and substitute ones that the mutator wanted changed.
fn pass_one_to_one<'a>(
    original: &'a Computation,
    mutation: fn(&'a Operation) -> Option<Operation>,
) -> Computation {
    Computation {
        operations: original
            .operations
            .iter()
            .map(|op| {
                if let Some(new_op) = mutation(op) {
                    new_op
                } else {
                    op.clone()
                }
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    use crate::execution::TestExecutor;

    #[test]
    /// Just a check that computation can be parsed and executed before any changes made
    fn test_basic_computation() -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Input() {arg_name = "x"} : () -> Float32Tensor @Host(alice)
        y = Input() {arg_name = "y"} : () -> Float32Tensor @Host(alice)
        sum = StdAdd(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @Host(alice)
        output = Output(sum): (Float32Tensor) -> Float32Tensor @Host(alice)
        "#;
        let comp: Computation = source.try_into()?;

        let exec = TestExecutor::default();
        let args = [
            ("x".to_string(), "[1.0]: Float32Tensor".try_into()?),
            ("y".to_string(), "[2.0]: Float32Tensor".try_into()?),
        ];

        let outputs = exec.run_computation(&comp, args.iter().cloned().collect())?;

        let sum = outputs.get("output").unwrap();
        let expected: Value = "[3.0]: Float32Tensor".try_into()?;
        assert_eq!(sum, &expected);

        Ok(())
    }

    /// A test that replaces StdAdd with StdMul. Then executes the modified computation.
    #[test]
    fn test_one_to_one_symbolic() -> std::result::Result<(), anyhow::Error> {
        let source = r#"x = Input() {arg_name = "x"} : () -> Float32Tensor @Host(alice)
        y = Input() {arg_name = "y"} : () -> Float32Tensor @Host(alice)
        sum = StdAdd(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @Host(alice)
        output = Output(sum): (Float32Tensor) -> Float32Tensor @Host(alice)
        "#;
        let original: Computation = source.try_into()?;
        let comp = pass_one_to_one(
            &original,
            replace_add_with_mul::<SymbolicOperandContext, SymbolicOneToOneGraphMutator>,
        );

        let exec = TestExecutor::default();
        let args = [
            ("x".to_string(), "[1.0]: Float32Tensor".try_into()?),
            ("y".to_string(), "[2.0]: Float32Tensor".try_into()?),
        ];

        let outputs = exec.run_computation(&comp, args.iter().cloned().collect())?;

        let sum = outputs.get("output").unwrap();
        let expected: Value = "[2.0]: Float32Tensor".try_into()?;
        assert_eq!(sum, &expected);

        Ok(())
    }

    /// A test that grabs the StdAdd operation, and returns a lambda for it, which inside is not an addition, but multiplication instead.
    #[test]
    fn test_one_to_one_concrete() -> std::result::Result<(), anyhow::Error> {
        let source =
            r#"sum = StdAdd(x, y): (Float32Tensor, Float32Tensor) -> Float32Tensor @Host(alice)"#;
        let original: Computation = source.try_into()?;

        let lambda = replace_add_with_mul::<ConcreteOperandContext, ConcreteOneToOneGraphMutator>(
            &original.operations[0],
        )
        .unwrap();

        // Now we can use that "compiled" lambda as we please
        let sum = lambda(vec![
            "[3.0]: Float32Tensor".try_into()?,
            "[4.0]: Float32Tensor".try_into()?,
        ]);
        let expected: Value = "[12.0]: Float32Tensor".try_into()?;
        assert_eq!(sum, expected);

        Ok(())
    }
}
