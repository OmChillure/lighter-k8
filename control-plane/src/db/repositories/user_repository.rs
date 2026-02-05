use diesel::prelude::*;
use uuid::Uuid;

use super::super::schema::users;
use crate::db::models::{UpdateUser, User};
use crate::error::{AppError, Result};

pub struct UserRepository;

impl UserRepository {
    pub fn find_by_id(conn: &mut PgConnection, user_id: Uuid) -> Result<User> {
        users::table
            .find(user_id)
            .select(User::as_select())
            .first(conn)
            .map_err(AppError::from)
    }

    pub fn list_all(conn: &mut PgConnection) -> Result<Vec<User>> {
        users::table
            .select(User::as_select())
            .load(conn)
            .map_err(AppError::from)
    }

    pub fn find_by_username(conn: &mut PgConnection, user_username: &str) -> Result<User> {
        users::table
            .filter(users::username.eq(user_username))
            .select(User::as_select())
            .first(conn)
            .map_err(AppError::from)
    }

    pub fn find_by_apikey(conn: &mut PgConnection, user_apikey: &str) -> Result<User> {
        users::table
            .filter(users::api_key.eq(user_apikey))
            .select(User::as_select())
            .first(conn)
            .map_err(AppError::from)
    }


    /// Create a new user
    pub fn create(
        conn: &mut PgConnection,
        id: Uuid,
        username: String,
        api_key: String,
        account_index: i32,
        api_key_index: i32,
    ) -> Result<User> {
        #[derive(Insertable)]
        #[diesel(table_name = users)]
        struct NewUser {
            id: Uuid,
            username: String,
            api_key: String,
            account_index: i32,
            api_key_index: i32,
        }

        let new_user = NewUser {
            id,
            username,
            api_key,
            account_index,
            api_key_index,
        };

        diesel::insert_into(users::table)
            .values(&new_user)
            .returning(User::as_returning())
            .get_result(conn)
            .map_err(AppError::from)
    }

    /// Update a user
    pub fn update(conn: &mut PgConnection, user_id: Uuid, update_data: UpdateUser) -> Result<User> {
        diesel::update(users::table.find(user_id))
            .set(&update_data)
            .returning(User::as_returning())
            .get_result(conn)
            .map_err(AppError::from)
    }

    /// Delete a user
    pub fn delete(conn: &mut PgConnection, user_id: Uuid) -> Result<usize> {
        diesel::delete(users::table.find(user_id))
            .execute(conn)
            .map_err(AppError::from)
    }
}
