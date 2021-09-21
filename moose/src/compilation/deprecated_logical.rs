use crate::computation::*;

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
        Ty::Tensor(InnerTy::Float32) => Ty::HostFloat32Tensor,
        Ty::Tensor(InnerTy::Float64) => Ty::HostFloat64Tensor,
        Ty::Tensor(InnerTy::Fixed64 { precision: _ }) => Ty::HostFixed64Tensor,
        Ty::Tensor(InnerTy::Fixed128 { precision: _ }) => Ty::HostFixed128Tensor,
        _ => ty,
    }
}

fn lower_op(op: &Operation) -> Operation {
    match op.kind {
        Operator::AtLeast2D(ref i) => Operation {
            name: op.name.clone(),
            kind: HostAtLeast2DOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                to_column_vector: i.to_column_vector,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Mean(ref i) => Operation {
            name: op.name.clone(),
            kind: HostMeanOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                axis: i.axis,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Sum(ref i) => Operation {
            name: op.name.clone(),
            kind: HostSumOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                axis: i.axis,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Concat(ref i) => Operation {
            name: op.name.clone(),
            kind: HostConcatOp {
                sig: Signature::binary(
                    lower_ty(i.sig.arg(0).unwrap()),
                    lower_ty(i.sig.arg(1).unwrap()),
                    lower_ty(i.sig.ret()),
                ),
                axis: i.axis,
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Add(ref i) => Operation {
            name: op.name.clone(),
            kind: HostAddOp {
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
        Operator::Sub(ref i) => Operation {
            name: op.name.clone(),
            kind: HostSubOp {
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
        Operator::Mul(ref i) => Operation {
            name: op.name.clone(),
            kind: HostMulOp {
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
        Operator::Dot(ref i) => Operation {
            name: op.name.clone(),
            kind: HostDotOp {
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
        Operator::Div(ref i) => Operation {
            name: op.name.clone(),
            kind: HostDivOp {
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
        Operator::ExpandDims(ref i) => Operation {
            name: op.name.clone(),
            kind: HostExpandDimsOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                axis: i.axis.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::RingFixedpointEncode(ref i) => Operation {
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
        Operator::RingFixedpointDecode(ref i) => Operation {
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
        Operator::Shape(ref i) => Operation {
            name: op.name.clone(),
            kind: ShapeOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Slice(ref i) => Operation {
            name: op.name.clone(),
            kind: HostSliceOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
                slice: i.slice.clone(),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Ones(ref i) => Operation {
            name: op.name.clone(),
            kind: HostOnesOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Identity(ref i) => Operation {
            name: op.name.clone(),
            kind: IdentityOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Transpose(ref i) => Operation {
            name: op.name.clone(),
            kind: HostTransposeOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Inverse(ref i) => Operation {
            name: op.name.clone(),
            kind: HostInverseOp {
                sig: Signature::unary(lower_ty(i.sig.arg(0).unwrap()), lower_ty(i.sig.ret())),
            }
            .into(),
            inputs: op.inputs.clone(),
            placement: op.placement.clone(),
        },
        Operator::Load(ref i) => Operation {
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
        Operator::Save(ref i) => Operation {
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
        Operator::Send(ref i) => Operation {
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
        Operator::Receive(ref i) => Operation {
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

        _ => op.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_computation::ToTextual;
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
