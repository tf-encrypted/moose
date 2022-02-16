use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct SessionConfig {
    pub(super) computation: ComputationConfig,
    pub(super) roles: Vec<RoleConfig>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ComputationConfig {
    pub(super) path: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct RoleConfig {
    pub(super) name: String,
    pub(super) endpoint: String,
}
