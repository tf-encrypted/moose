use crate::computation::*;
use crate::logical::TensorDType;

/// The pass replaces the logical level ops with their Host counterparts.
///
/// It is used to process computations deserialized from python-traced-and-compiled computations.
/// Once we full switch to Rust compiling of python-traced computations, this pass should be deleted.
pub fn deprecated_logical_lowering(comp: Computation) -> anyhow::Result<Computation> {
    let operations = comp.operations.iter().map(lower_op).collect();
    Ok(Computation { operations })
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
            kind: AtLeast2DOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                to_column_vector: i.to_column_vector,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Mean(ref i)) => Operation {
            name: op.name.clone(),
            kind: MeanOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                axis: i.axis,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Add(ref i)) => Operation {
            name: op.name.clone(),
            kind: AddOp {
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
            kind: SubOp {
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
            kind: MulOp {
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
            kind: DotOp {
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
            kind: DivOp {
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
            kind: ExpandDimsOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                axis: i.axis.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::RingFixedpointEncode(ref i)) => Operation {
            name: op.name.clone(),
            kind: RingFixedpointEncodeOp {
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
            kind: RingFixedpointDecodeOp {
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
            kind: ShapeOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Slice(ref i)) => Operation {
            name: op.name.clone(),
            kind: SliceOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                slice: i.slice.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Ones(ref i)) => Operation {
            name: op.name.clone(),
            kind: OnesOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Identity(ref i)) => Operation {
            name: op.name.clone(),
            kind: IdentityOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Transpose(ref i)) => Operation {
            name: op.name.clone(),
            kind: TransposeOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Inverse(ref i)) => Operation {
            name: op.name.clone(),
            kind: InverseOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Load(ref i)) => Operation {
            name: op.name.clone(),
            kind: LoadOp {
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
            kind: SaveOp {
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
            kind: SendOp {
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
            kind: ReceiveOp {
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
            kind: InputOp {
                sig: Signature::nullary(lower_ty(i.sig.ret())),
                arg_name: i.arg_name.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Output(ref i)) => Operation {
            name: op.name.clone(),
            kind: OutputOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        (Placement::Host(_), Operator::Constant(ref i)) => Operation {
            name: op.name.clone(),
            kind: ConstantOp {
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
        mul = Mul: (Tensor<Float64>, Tensor<Float64>) -> Tensor<Float64> (x, y) @Host(alice)
        save = Save: (HostString, Tensor<Float64>) -> HostUnit (constant_0, mean) @Host(alice)
        "#;

        let comp = deprecated_logical_lowering(source.try_into()?)?.to_textual();
        // The computation should now contain the modified type information
        assert!(comp.contains(
            "mul = Mul: (HostFloat64Tensor, HostFloat64Tensor) -> HostFloat64Tensor (x, y) @Host(alice)"
        ));
        assert!(comp.contains(
            "save = Save: (HostString, HostFloat64Tensor) -> HostUnit (constant_0, mean) @Host(alice)"
        ));
        Ok(())
    }
}
