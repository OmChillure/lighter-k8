use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use uuid::Uuid;

use crate::{
    db::{
        models::{CreateUser, UpdateUser, User},
    },
    error::Result,
    services::UserService,
    state::AppState,
};

pub async fn health_check() -> Result<Json<serde_json::Value>> {
    Ok(Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })))
}

pub async fn list_users(State(state): State<AppState>) -> Result<Json<Vec<User>>> {
    let users = UserService::list_users(&state.pool)?;
    Ok(Json(users))
}

pub async fn get_user(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
) -> Result<Json<User>> {
    let user = UserService::get_user(&state.pool, user_id).await?;
    Ok(Json(user))
}

pub async fn get_user_by_apikey(
    State(state): State<AppState>,
    Path(apikey): Path<String>,
) -> Result<Json<User>> {
    let user = UserService::get_user_by_apikey(&state.pool, apikey).await?;
    Ok(Json(user))
}

/// Create a new user and automatically spawn agent pod
pub async fn create_user(
    State(state): State<AppState>,
    Json(payload): Json<CreateUser>,
) -> Result<(StatusCode, Json<serde_json::Value>)> {
    let user = UserService::create_user(&state.pool, payload)?;
    
    // Automatically spawn agent pod for the new user
    let pod_result = state.pod_service.spawn_agent(&user).await;
    
    let response = match pod_result {
        Ok(pod_name) => serde_json::json!({
            "user": user,
            "agent_status": "started",
            "pod_name": pod_name
        }),
        Err(e) => {
            // Log the error but still return the user since user creation succeeded
            tracing::warn!("User created successfully but failed to spawn agent: {}", e);
            serde_json::json!({
                "user": user,
                "agent_status": "failed",
                "error": format!("Failed to spawn agent: {}", e)
            })
        }
    };
    
    Ok((StatusCode::CREATED, Json(response)))
}

/// Update user
pub async fn update_user(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
    Json(payload): Json<UpdateUser>,
) -> Result<Json<User>> {
    let user = UserService::update_user(&state.pool, user_id, payload)?;
    Ok(Json(user))
}

/// Delete user and stop agent pod
pub async fn delete_user(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
) -> Result<StatusCode> {
    // First try to stop the agent pod
    if let Err(e) = state.pod_service.stop_agent(&user_id).await {
        tracing::warn!("Failed to stop agent pod for user {}: {}", user_id, e);
    }
    
    // Then delete the user
    UserService::delete_user(&state.pool, user_id)?;
    Ok(StatusCode::NO_CONTENT)
}
