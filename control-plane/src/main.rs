mod config;
mod db;
mod error;
mod handlers;
mod middleware;
mod routes;
mod services;
mod state;

use axum::middleware as axum_middleware;
use config::Config;
use std::net::SocketAddr;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let config = Config::from_env()?;
    tracing::info!("Configuration loaded successfully");

    let pool = db::create_pool(&config.database.url, config.database.max_connections);
    tracing::info!("Database pool created");

    let mut conn = pool.get()?;
    db::run_migrations(&mut conn);
    tracing::info!("Database migrations completed");

    let pod_service = std::sync::Arc::new(services::PodService::new().await?);
    tracing::info!("PodService initialized");

    let app_state = state::AppState { pool, pod_service };

    let app = routes::create_routes(app_state)
        .layer(middleware::cors_layer())
        .layer(TraceLayer::new_for_http())
        .layer(axum_middleware::from_fn(middleware::logging_middleware))
        .layer(axum_middleware::from_fn(middleware::request_id_middleware));

    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!(
        "🚀 Server running at http://{}:{}",
        config.server.host,
        config.server.port
    );
    tracing::info!(
        "📊 Health check: http://{}:{}/health",
        config.server.host,
        config.server.port
    );
    tracing::info!(
        "🔌 API endpoint: http://{}:{}/api/v1",
        config.server.host,
        config.server.port
    );

    axum::serve(listener, app).await?;

    Ok(())
}

fn init_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "axum_api=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}
