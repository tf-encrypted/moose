use crate::computation::*;
use crate::logical::TensorShape;

/// The pass replaces the HostShape on the logical level with its Shape<Host> counterpart.
pub fn deprecated_shape_support(comp: Computation) -> anyhow::Result<Computation> {
    let mut operations = comp.operations.clone();
    let mut changes_made = false; // A flag to let the SliceOp know if it needs to alter its types as well.
    for op in operations.iter_mut() {
        // Recognize ops that have logical level kernels and may see a shape of a wrong type.
        match op.kind {
            Operator::Shape(ShapeOp {
                sig:
                    Signature::Unary(UnarySignature {
                        arg0: Ty::Tensor(_),
                        ret: ref mut ret_ty,
                    }),
            }) if *ret_ty == Ty::HostShape => {
                *ret_ty = Ty::Shape(TensorShape::Host);
                changes_made = true;
            }
            Operator::Ones(OnesOp {
                sig:
                    Signature::Unary(UnarySignature {
                        arg0: ref mut arg0_ty,
                        ret: Ty::Tensor(_),
                    }),
            }) if *arg0_ty == Ty::HostShape => {
                *arg0_ty = Ty::Shape(TensorShape::Host);
            }
            Operator::Slice(SliceOp {
                sig:
                    Signature::Unary(UnarySignature {
                        arg0: Ty::HostShape,
                        ret: Ty::HostShape,
                    }),
                ..
            }) if changes_made => {
                *op.kind.sig_mut() =
                    Signature::unary(Ty::Shape(TensorShape::Host), Ty::Shape(TensorShape::Host));
            }
            _ => {}
        };
    }
    Ok(Computation { operations })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textual::ToTextual;
    use std::convert::TryInto;

    #[test]
    fn test_host_shape_replace() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        shape_0 = Shape: (Tensor<Fixed128(24, 40)>) -> HostShape (decrypt_0) @Host(bob)
        "#;

        let comp = deprecated_shape_support(source.try_into()?)?.to_textual();
        // The computation should now contain the modified type information
        assert_eq!(
            comp,
            "shape_0 = Shape: (Tensor<Fixed128(24, 40)>) -> Shape<Host> (decrypt_0) @Host(bob)"
        );
        Ok(())
    }
}
