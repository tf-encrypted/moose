use crate::computation::{HostPlacement, InputOp, ReplicatedPlacement};
use crate::error::Result;
use crate::kernels::{PlacementInput, Session};
use crate::replicated::AbstractReplicatedRingTensor;

impl InputOp {
    pub(crate) fn replicated_ring_kernel<S: Session, HostRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        arg_name: String,
    ) -> Result<AbstractReplicatedRingTensor<HostRingT>>
    where
        HostPlacement: PlacementInput<S, HostRingT>,
    {
        // TODO standardize this arg_name format for shares
        let lift_name = |name, player_ix, share_ix| {
            format!("{0}_player{1}_share{2}", name, player_ix, share_ix)
        };
        let (p0, p1, p2) = plc.host_placements();
        let in00 = p0.input(sess, lift_name(arg_name.clone(), 0, 0));
        let in01 = p0.input(sess, lift_name(arg_name.clone(), 0, 1));
        let in10 = p1.input(sess, lift_name(arg_name.clone(), 1, 0));
        let in11 = p1.input(sess, lift_name(arg_name.clone(), 1, 1));
        let in20 = p2.input(sess, lift_name(arg_name.clone(), 2, 0));
        let in21 = p2.input(sess, lift_name(arg_name, 2, 1));
        Ok(AbstractReplicatedRingTensor {
            shares: [[in00, in01], [in10, in11], [in20, in21]],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computation::SessionId;
    use crate::host::HostRing64Tensor;
    use crate::replicated::ReplicatedRing64Tensor;
    use ndarray::{array, IxDyn};

    use crate::kernels::{PlacementReveal, PlacementShareSetup, SyncSession};

    #[test]
    fn test_rep_input_op() {
        let alice = HostPlacement {
            owner: "alice".into(),
        };
        let rep = ReplicatedPlacement {
            owners: ["alice".into(), "bob".into(), "carole".into()],
        };

        // Create replicated input tensor in a previous session
        let sess0 = SyncSession::default();
        let setup = (*sess0.replicated_setup(&rep)).clone();
        let x = HostRing64Tensor::from_raw_plc(
            array![1u64, 2, 3].into_dimensionality::<IxDyn>().unwrap(),
            alice.clone(),
        );
        let x_shared = rep.share(&sess0, &setup, &x);

        // Populate test session args with shares of x
        let arg_name = "x".to_string();
        let lift_name = |name, player_ix, share_ix| {
            format!("{0}_player{1}_share{2}", name, player_ix, share_ix)
        };
        let mut new_args = std::collections::HashMap::new();
        for i in 0..3 {
            for j in 0..2 {
                new_args.insert(
                    lift_name(arg_name.clone(), i, j),
                    x_shared.shares[i][j].clone().into(),
                );
            }
        }

        // Test input op
        let test_sess = SyncSession::new(SessionId::random(), new_args, Default::default());
        let y: ReplicatedRing64Tensor = rep.input(&test_sess, arg_name);
        let z = alice.reveal(&test_sess, &y);
        assert_eq!(x, z)
    }
}
