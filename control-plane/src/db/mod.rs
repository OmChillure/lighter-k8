use diesel::prelude::*;
use diesel::r2d2::{ConnectionManager, Pool};
use std::time::Duration;

pub mod models;
pub mod repositories;
pub mod schema;

pub type DbPool = Pool<ConnectionManager<PgConnection>>;

/// Create a database connection pool
pub fn create_pool(database_url: &str, max_connections: u32) -> DbPool {
    let manager = ConnectionManager::<PgConnection>::new(database_url);

    Pool::builder()
        .max_size(max_connections)
        .connection_timeout(Duration::from_secs(30))
        .build(manager)
        .expect("Failed to create database pool")
}

/// Run pending migrations
pub fn run_migrations(conn: &mut PgConnection) {
    use diesel_migrations::{embed_migrations, EmbeddedMigrations, MigrationHarness};

    pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");

    conn.run_pending_migrations(MIGRATIONS)
        .expect("Failed to run migrations");
}
