use chrono::NaiveDateTime;
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use super::super::schema::users;

/// User model for database
#[derive(Debug, Queryable, Selectable, Serialize)]
#[diesel(table_name = users)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub api_key: String,
    pub account_index: i32,
    pub api_key_index: i32,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
}

#[derive(Debug, Deserialize, Clone, AsChangeset,Validate)]
#[diesel(table_name = users)]
pub struct UpdateUser {
    pub username: Option<String>,
    pub api_key: Option<String>,
    pub account_index: Option<i32>,
    pub api_key_index: Option<i32>,
}

#[derive(Debug, Deserialize, Clone, AsChangeset, Validate)]
#[diesel(table_name = users)]
pub struct CreateUser {
    pub username: String,
    pub api_key: String,
    pub account_index: i32,
    pub api_key_index: i32,
}