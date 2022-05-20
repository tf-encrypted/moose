use moose::prelude::{Computation, Identity, Role, SessionId};
use std::collections::HashMap;

pub struct GrpcMooseRuntime {
    role_assignments: HashMap<Role, Identity>,
}

impl GrpcMooseRuntime {
    pub fn new(role_assignments: HashMap<Role, Identity>) -> GrpcMooseRuntime {
        GrpcMooseRuntime { role_assignments }
    }

    pub fn launch_computation(&self, session_id: &SessionId, comp: &Computation) {
        // TODO(Morten) send session_id, comp, and self.role_assignments across to every identity in self.role_assignments
    }

    pub fn abort_computation(&self, session_id: &SessionId) {
        // TODO(Morten) tell every identity in self.role_assignments to abort session_id
    }
}
