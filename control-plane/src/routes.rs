use axum::{routing::get, Router};

use crate::{handlers, state::AppState};

pub fn create_routes(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health_check))
        .nest("/api", api_routes())
        .with_state(state)
}

fn api_routes() -> Router<AppState> {
    Router::new().nest("/v1", v1_routes())
}

fn v1_routes() -> Router<AppState> {
    Router::new()
        .nest("/users", user_routes())
        .nest("/agents", agent_routes())
}

fn user_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::list_users).post(handlers::create_user))
        .route(
            "/:id",
            get(handlers::get_user)
                .put(handlers::update_user)
                .delete(handlers::delete_user),
        )
        .route("/key/:apikey", get(handlers::get_user_by_apikey))
}

fn agent_routes() -> Router<AppState> {
    use axum::routing::{post, get};
    Router::new()
        .route("/:user_id/start", post(handlers::start_agent))
        .route("/:user_id/stop", post(handlers::stop_agent))
        .route("/:user_id/status", get(handlers::get_agent_status))
        .route("/", get(handlers::list_agents))
}
