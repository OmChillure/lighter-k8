use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde_json::json;
use uuid::Uuid;

use crate::services::UserService;
use crate::state::AppState;


pub async fn start_agent(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
) -> impl IntoResponse {
    let user = match UserService::get_user(&state.pool, user_id).await {
        Ok(user) => {
            user
        },
        Err(_e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "User not found"})),
            )
                .into_response()
        }
    };

    match state.pod_service.spawn_agent(&user).await {
        Ok(pod_name) => (
            StatusCode::OK,
            Json(json!({
                "status": "started",
                "pod_name": pod_name,
                "user_id": user.id
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": format!("Failed to spawn agent: {}", e)
            })),
        )
            .into_response(),
    }
}

pub async fn stop_agent(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
) -> impl IntoResponse {
    // 1. Stop Agent
    match state.pod_service.stop_agent(&user_id).await {
        Ok(_) => (
            StatusCode::OK,
            Json(json!({
                "status": "stopped",
                "user_id": user_id
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": format!("Failed to stop agent: {}", e)
            })),
        )
            .into_response(),
    }
}

pub async fn get_agent_status(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
) -> impl IntoResponse {
    match state.pod_service.get_agent_status(&user_id).await {
        Ok(Some(status)) => (
            StatusCode::OK,
            Json(json!({
                "user_id": user_id,
                "status": status,
                "running": status == "Running"
            })),
        ).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "user_id": user_id,
                "status": "not_found",
                "running": false
            })),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": format!("Failed to get agent status: {}", e)
            })),
        ).into_response(),
    }
}

pub async fn list_agents(
    State(state): State<AppState>,
) -> impl IntoResponse {
    match state.pod_service.list_agent_pods().await {
        Ok(agents) => {
            let agent_list: Vec<_> = agents.into_iter().map(|(user_id, pod_name, status)| {
                json!({
                    "user_id": user_id,
                    "pod_name": pod_name,
                    "status": status,
                    "running": status == "Running"
                })
            }).collect();
            
            (
                StatusCode::OK,
                Json(json!({
                    "agents": agent_list,
                    "count": agent_list.len()
                })),
            ).into_response()
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": format!("Failed to list agents: {}", e)
            })),
        ).into_response(),
    }
}
