use crate::computation::*;
use crate::logical::TensorDType;

/// The pass replaces the logical level ops with their Host counterparts.
///
/// It is used to process computations deserialized from python-traced-and-compiled computations.
/// Once we full switch to Rust compiling of python-traced computations, this pass should be deleted.
pub fn deprecated_logical_lowering(comp: &Computation) -> anyhow::Result<Option<Computation>> {
    let operations = comp.operations.iter().map(lower_op).collect();
    Ok(Some(Computation { operations }))
}

fn lower_ty(ty: Ty) -> Ty {
    match ty {
        Ty::Tensor(TensorDType::Float32) => Ty::HostFloat32Tensor,
        Ty::Tensor(TensorDType::Float64) => Ty::HostFloat64Tensor,
        Ty::Tensor(TensorDType::Fixed64 { .. }) => Ty::HostFixed64Tensor,
        Ty::Tensor(TensorDType::Fixed128 { .. }) => Ty::HostFixed128Tensor,
        _ => ty,
    }
}

fn lower_op(op: &Operation) -> Operation {
    match (&op.placement, &op.kind) {
        (Placement::Host(_), Operator::AtLeast2D(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostAtLeast2D {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                to_column_vector: i.to_column_vector,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Mean(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostMean {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                axis: i.axis,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Sum(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostSum {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                axis: i.axis,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Concat(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostConcat {
                sig: Signature::variadic(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                axis: i.axis,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Add(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostAdd {
                sig: Signature::binary(
                    lower_ty(i.sig.arg(0).unwrap()),
                    lower_ty(i.sig.arg(1).unwrap()),
                    lower_ty(i.sig.ret()),
                ),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Sub(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostSub {
                sig: Signature::binary(
                    lower_ty(i.sig.arg(0).unwrap()),
                    lower_ty(i.sig.arg(1).unwrap()),
                    lower_ty(i.sig.ret()),
                ),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Mul(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostMul {
                sig: Signature::binary(
                    lower_ty(i.sig.arg(0).unwrap()),
                    lower_ty(i.sig.arg(1).unwrap()),
                    lower_ty(i.sig.ret()),
                ),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Dot(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostDot {
                sig: Signature::binary(
                    lower_ty(i.sig.arg(0).unwrap()),
                    lower_ty(i.sig.arg(1).unwrap()),
                    lower_ty(i.sig.ret()),
                ),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Div(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostDiv {
                sig: Signature::binary(
                    lower_ty(i.sig.arg(0).unwrap()),
                    lower_ty(i.sig.arg(1).unwrap()),
                    lower_ty(i.sig.ret()),
                ),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::ExpandDims(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostExpandDims {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                axis: i.axis.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::RingFixedpointEncode(ref i)) => Operation {
            name: op.name.clone(),
            kind: RingFixedpointEncode {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                scaling_base: i.scaling_base,
                scaling_exp: i.scaling_exp,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::RingFixedpointDecode(ref i)) => Operation {
            name: op.name.clone(),
            kind: RingFixedpointDecode {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                scaling_base: i.scaling_base,
                scaling_exp: i.scaling_exp,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Shape(ref i)) => Operation {
            name: op.name.clone(),
            kind: crate::computation::Shape {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Slice(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostSlice {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                slice: i.slice.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Ones(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostOnes {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Identity(ref i)) => Operation {
            name: op.name.clone(),
            kind: Identity {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Transpose(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostTranspose {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Inverse(ref i)) => Operation {
            name: op.name.clone(),
            kind: HostInverse {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Load(ref i)) => Operation {
            name: op.name.clone(),
            kind: Load {
                sig: Signature::binary(
                    lower_ty(i.sig.arg(0).unwrap()),
                    lower_ty(i.sig.arg(1).unwrap()),
                    lower_ty(i.sig.ret()),
                ),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Save(ref i)) => Operation {
            name: op.name.clone(),
            kind: Save {
                sig: Signature::binary(
                    lower_ty(i.sig.arg(0).unwrap()),
                    lower_ty(i.sig.arg(1).unwrap()),
                    lower_ty(i.sig.ret()),
                ),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Send(ref i)) => Operation {
            name: op.name.clone(),
            kind: Send {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                rendezvous_key: i.rendezvous_key.clone(),
                receiver: i.receiver.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Receive(ref i)) => Operation {
            name: op.name.clone(),
            kind: Receive {
                sig: Signature::nullary(lower_ty(i.sig.ret())),
                rendezvous_key: i.rendezvous_key.clone(),
                sender: i.sender.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Input(ref i)) => Operation {
            name: op.name.clone(),
            kind: Input {
                sig: Signature::nullary(lower_ty(i.sig.ret())),
                arg_name: i.arg_name.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Output(ref i)) => Operation {
            name: op.name.clone(),
            kind: Output {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::ConstantOp(ref i)) => Operation {
            name: op.name.clone(),
            kind: crate::computation::ConstantOp {
                sig: Signature::nullary(lower_ty(i.sig.ret())),
                value: i.value.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },

        _ => op.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textual::ToTextual;
    use std::convert::TryInto;

    #[test]
    fn test_all_on_one_host() -> std::result::Result<(), anyhow::Error> {
        let source = r#"
        mul = Mul: (Tensor, Tensor) -> Tensor (x, y) @Host(alice)
        save = Save: (String, Tensor) -> Unit (constant_0, mean) @Host(alice)
        "#;

        let comp = deprecated_logical_lowering(&source.try_into()?)?
            .unwrap()
            .to_textual();
        // The computation should now contain the modified type information
        assert!(comp.contains(
            "mul = HostMul: (Float64Tensor, Float64Tensor) -> Float64Tensor (x, y) @Host(alice)"
        ));
        assert!(comp.contains(
            "save = Save: (String, Float64Tensor) -> Unit (constant_0, mean) @Host(alice)"
        ));
        Ok(())
    }
}
