use uuid::Uuid;
use validator::Validate;

use crate::db::models::{CreateUser, UpdateUser, User};
use crate::db::repositories::{UserRepository};
use crate::db::DbPool;
use crate::error::{AppError, Result};

pub struct UserService;

impl UserService {
    /// Get user by ID
    pub async fn get_user(pool: &DbPool, user_id: Uuid) -> Result<User> {
        let mut conn = pool
            .get()
            .map_err(|e| AppError::Internal(format!("Failed to get DB connection: {}", e)))?;

        UserRepository::find_by_id(&mut conn, user_id)
    }

    pub async fn get_user_by_apikey(pool: &DbPool, user_apikey: String) -> Result<User> {
        let mut conn = pool
            .get()
            .map_err(|e| AppError::Internal(format!("Failed to get DB connection: {}", e)))?;

        UserRepository::find_by_apikey(&mut conn, &user_apikey)
    }

    /// List all users
    pub fn list_users(pool: &DbPool) -> Result<Vec<User>> {
        let mut conn = pool
            .get()
            .map_err(|e| AppError::Internal(format!("Failed to get DB connection: {}", e)))?;

        UserRepository::list_all(&mut conn)
    }

    /// Create a new user with validation and password hashing
    pub fn create_user(pool: &DbPool, input: CreateUser) -> Result<User> {
        // Validate input
        input
            .validate()
            .map_err(|e| AppError::Validation(e.to_string()))?;

        let mut conn = pool
            .get()
            .map_err(|e| AppError::Internal(format!("Failed to get DB connection: {}", e)))?;

        if let Ok(_) = UserRepository::find_by_username(&mut conn, &input.username) {
            return Err(AppError::BadRequest("Username already exists".to_string()));
        }

        // Create user
        UserRepository::create(
            &mut conn,
            Uuid::new_v4(),
            input.username,
            input.api_key,
            input.account_index,
            input.api_key_index,
        )
    }

    /// Update user
    pub fn update_user(pool: &DbPool, user_id: Uuid, input: UpdateUser) -> Result<User> {
        input
            .validate()
            .map_err(|e| AppError::Validation(e.to_string()))?;

        let mut conn = pool
            .get()
            .map_err(|e| AppError::Internal(format!("Failed to get DB connection: {}", e)))?;

        // Check if email is being changed and already exists
        if let Some(ref username) = input.username {
            if let Ok(existing_user) = UserRepository::find_by_username(&mut conn, username) {
                if existing_user.id != user_id {
                    return Err(AppError::BadRequest("Username already exists".to_string()));
                }
            }
        }

        UserRepository::update(&mut conn, user_id, input)
    }

    /// Delete user
    pub fn delete_user(pool: &DbPool, user_id: Uuid) -> Result<()> {
        let mut conn = pool
            .get()
            .map_err(|e| AppError::Internal(format!("Failed to get DB connection: {}", e)))?;

        let deleted = UserRepository::delete(&mut conn, user_id)?;

        if deleted > 0 {
            Ok(())
        } else {
            Err(AppError::NotFound("User not found".to_string()))
        }
    }
}
