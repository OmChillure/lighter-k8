use crate::db::DbPool;
use crate::services::PodService;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub pool: DbPool,
    pub pod_service: Arc<PodService>,
}
