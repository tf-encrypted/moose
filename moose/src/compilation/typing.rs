use crate::computation::*;
use crate::logical::TensorDType;
use petgraph::Direction;
use std::collections::HashMap;

/// Updates the operators such that the type information is inferred by one-hop check, without any recursive graph traversal.
pub(crate) fn update_types_one_hop(comp: &Computation) -> anyhow::Result<Computation> {
    let mut operations = comp.operations.clone();
    let graph = comp.as_graph();

    for n in graph.node_indices() {
        // Prepare the raw data for the signature computation
        let inputs = &comp.operations[graph[n].index].inputs;
        let types: HashMap<&String, Ty> = graph
            .neighbors_directed(n, Direction::Incoming)
            .map(|i| {
                (
                    graph[i].op_name,
                    comp.operations[graph[i].index].kind.sig().ret(),
                )
            })
            .collect();
        let ret = comp.operations[graph[n].index].kind.sig().ret();

        let find_type = |i: usize| -> anyhow::Result<Ty> {
            match types.get(&inputs[i]) {
                Some(ty) => Ok(*ty),
                _ => Err(anyhow::anyhow!(
                    "Could not find type of input {}",
                    inputs[i]
                )),
            }
        };

        // Compute the new signature from the graph
        let new_sig = match inputs.len() {
            0 => Signature::nullary(ret),
            1 => Signature::unary(find_type(0)?, ret),
            2 => Signature::binary(find_type(0)?, find_type(1)?, ret),
            3 => Signature::ternary(find_type(0)?, find_type(1)?, find_type(2)?, ret),
            n => {
                assert!((0..n).all(|i| find_type(i).ok() == find_type(0).ok()));
                Signature::variadic(find_type(0)?, ret)
            }
        };

        // Update the existing signature with it.
        operations[graph[n].index].kind.sig_mut().merge(new_sig)?;
    }
    Ok(Computation { operations })
}

impl Signature {
    fn merge(&mut self, another: Signature) -> anyhow::Result<()> {
        match (self, &another) {
            (Signature::Nullary(s), Signature::Nullary(o)) => s.merge(o),
            (Signature::Unary(s), Signature::Unary(o)) => s.merge(o),
            (Signature::Binary(s), Signature::Binary(o)) => s.merge(o),
            (Signature::Ternary(s), Signature::Ternary(o)) => s.merge(o),
            (Signature::Variadic(s), o) => s.merge(o),

            (Signature::Nullary(s), o) => Err(anyhow::anyhow!(
                "Cannot merge {:?} with an incompatible signature {:?}",
                s,
                o
            )),
            (Signature::Unary(s), o) => Err(anyhow::anyhow!(
                "Cannot merge {:?} with an incompatible signature {:?}",
                s,
                o
            )),
            (Signature::Binary(s), o) => Err(anyhow::anyhow!(
                "Cannot merge {:?} with an incompatible signature {:?}",
                s,
                o
            )),
            (Signature::Ternary(s), o) => Err(anyhow::anyhow!(
                "Cannot merge {:?} with an incompatible signature {:?}",
                s,
                o
            )),
        }
    }
}

impl NullarySignature {
    fn merge(&mut self, another: &NullarySignature) -> anyhow::Result<()> {
        if let Some(new_type) = self.ret.merge(&another.ret) {
            self.ret = new_type;
        }
        Ok(())
    }
}

impl UnarySignature {
    fn merge(&mut self, another: &UnarySignature) -> anyhow::Result<()> {
        if let Some(new_type) = self.arg0.merge(&another.arg0) {
            self.arg0 = new_type;
        }
        if let Some(new_type) = self.ret.merge(&another.ret) {
            self.ret = new_type;
        }
        Ok(())
    }
}

impl BinarySignature {
    fn merge(&mut self, another: &BinarySignature) -> anyhow::Result<()> {
        if let Some(new_type) = self.arg0.merge(&another.arg0) {
            self.arg0 = new_type;
        }
        if let Some(new_type) = self.arg1.merge(&another.arg1) {
            self.arg1 = new_type;
        }
        if let Some(new_type) = self.ret.merge(&another.ret) {
            self.ret = new_type;
        }
        Ok(())
    }
}

impl TernarySignature {
    fn merge(&mut self, another: &TernarySignature) -> anyhow::Result<()> {
        if let Some(new_type) = self.arg0.merge(&another.arg0) {
            self.arg0 = new_type;
        }
        if let Some(new_type) = self.arg1.merge(&another.arg1) {
            self.arg1 = new_type;
        }
        if let Some(new_type) = self.arg2.merge(&another.arg2) {
            self.arg2 = new_type;
        }
        if let Some(new_type) = self.ret.merge(&another.ret) {
            self.ret = new_type;
        }
        Ok(())
    }
}

impl VariadicSignature {
    fn merge(&mut self, another: &Signature) -> anyhow::Result<()> {
        match another {
            Signature::Variadic(sig) => {
                if let Some(new_type) = self.args.merge(&sig.args) {
                    self.args = new_type;
                }
                if let Some(new_type) = self.ret.merge(&sig.ret) {
                    self.ret = new_type;
                }
                Ok(())
            }
            Signature::Unary(sig) => {
                if self.args == sig.arg0 {
                    if let Some(new_type) = self.args.merge(&sig.arg0) {
                        self.args = new_type;
                    }
                }

                if let Some(new_type) = self.ret.merge(&sig.ret) {
                    self.ret = new_type;
                }
                Ok(())
            }
            Signature::Binary(sig) => {
                if self.args == sig.arg0 && self.args == sig.arg1 {
                    if let Some(new_type) = self.args.merge(&sig.arg0) {
                        self.args = new_type;
                    }

                    if let Some(new_type) = self.args.merge(&sig.arg1) {
                        self.args = new_type;
                    }
                }

                if let Some(new_type) = self.ret.merge(&sig.ret) {
                    self.ret = new_type;
                }

                Ok(())
            }
            Signature::Ternary(sig) => {
                if self.args == sig.arg0 && self.args == sig.arg1 && self.args == sig.arg2 {
                    if let Some(new_type) = self.args.merge(&sig.arg0) {
                        self.args = new_type;
                    }

                    if let Some(new_type) = self.args.merge(&sig.arg1) {
                        self.args = new_type;
                    }

                    if let Some(new_type) = self.args.merge(&sig.arg2) {
                        self.args = new_type;
                    }
                }

                if let Some(new_type) = self.ret.merge(&sig.ret) {
                    self.ret = new_type;
                }

                Ok(())
            }
            o => Err(anyhow::anyhow!(
                "Cannot merge {:?} with an incompatible signature {:?}",
                self,
                o
            )),
        }
    }
}

impl Ty {
    /// Merge type information.
    ///
    /// Returns `Some(new_type)` if a merge produced a new type.
    /// Otherwise returns None
    fn merge(&self, another: &Ty) -> Option<Ty> {
        match self {
            Ty::Unknown => Some(*another),
            // TODO: make sure another dtype is also a tensor
            Ty::Tensor(TensorDType::Unknown) => Some(*another),
            _ => None,
        }
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
        x = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        y = Constant{value=HostFloat32Tensor([[1.0, 2.0], [3.0, 4.0]])}: () -> HostFloat32Tensor @Host(alice)
        mul = Mul: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        dot = Dot: (HostFloat32Tensor, HostFloat32Tensor) -> HostFloat32Tensor (x, y) @Host(alice)
        mean = Mean{}: (HostFloat32Tensor) -> HostFloat32Tensor (dot) @Host(alice)
        constant_0 = Constant{value = HostString("regression_weights")}: () -> HostString () @Host(alice)
        save = Save: (HostString, Unknown) -> HostUnit (constant_0, mean) @Host(alice)
        "#;

        let comp = update_types_one_hop(&source.try_into()?)?
            .unwrap()
            .to_textual();
        // The computation should now contain the type information
        assert!(comp.contains(
            "save = Save: (HostString, HostFloat32Tensor) -> HostUnit (constant_0, mean) @Host(alice)"
        ));
        Ok(())
    }
}
