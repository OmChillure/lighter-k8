use crate::db::models::user::User;
use anyhow::{Context, Result};
use k8s_openapi::api::core::v1::{Container, EnvVar, Pod, PodSpec, ResourceRequirements};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use kube::api::{DeleteParams, PostParams};
use kube::{Api, Client};
use std::collections::BTreeMap;

pub struct PodService {
    client: Client,
    namespace: String,
    image_name: String,
}

impl PodService {
    pub async fn new() -> Result<Self> {
        let client = Client::try_default()
            .await
            .context("Failed to create K8s client")?;

        Ok(Self {
            client,
            namespace: std::env::var("KUBERNETES_NAMESPACE").unwrap_or_else(|_| "default".to_string()),
            image_name: std::env::var("AGENT_IMAGE_NAME").unwrap_or_else(|_| "omchilluree/trading-agent:latest".to_string()),
        })
    }

    pub async fn spawn_agent(&self, user: &User) -> Result<String> {
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);

        let pod_name = format!("agent-user-{}", user.id);

        let env_vars = vec![
            EnvVar {
                name: "LIGHTER_API_PRIVATE_KEY".to_string(),
                value: Some(user.api_key.clone()),
                ..Default::default()
            },
            EnvVar {
                name: "LIGHTER_API_KEY_INDEX".to_string(),
                value: Some(user.api_key_index.to_string()),
                ..Default::default()
            },
            EnvVar {
                name: "LIGHTER_ACCOUNT_INDEX".to_string(),
                value: Some(user.account_index.to_string()),
                ..Default::default()
            },
            EnvVar {
                name: "DATABASE_URL".to_string(),
                value: Some(std::env::var("DATABASE_URL").unwrap_or_default()),
                ..Default::default()
            },
            EnvVar {
                name: "USER_ID".to_string(),
                value: Some(user.id.to_string()),
                ..Default::default()
            },
            EnvVar {
                name: "USERNAME".to_string(),
                value: Some(user.username.clone()),
                ..Default::default()
            },
        ];

        let mut resource_limits = BTreeMap::new();
        resource_limits.insert("memory".to_string(), Quantity("512Mi".to_string()));
        resource_limits.insert("cpu".to_string(), Quantity("500m".to_string()));
        
        let mut resource_requests = BTreeMap::new();
        resource_requests.insert("memory".to_string(), Quantity("256Mi".to_string()));
        resource_requests.insert("cpu".to_string(), Quantity("200m".to_string()));

        let pod = Pod {
            metadata: ObjectMeta {
                name: Some(pod_name.clone()),
                labels: Some(BTreeMap::from([
                    ("app".to_string(), "scalping-agent".to_string()),
                    ("user_id".to_string(), user.id.to_string()),
                    ("username".to_string(), user.username.clone()),
                    ("component".to_string(), "trading-agent".to_string()),
                    ("managed-by".to_string(), "control-plane".to_string()),
                ])),
                annotations: Some(BTreeMap::from([
                    ("created-by".to_string(), "control-plane".to_string()),
                    ("user-api-key-index".to_string(), user.api_key_index.to_string()),
                    ("user-account-index".to_string(), user.account_index.to_string()),
                ])),
                ..Default::default()
            },
            spec: Some(PodSpec {
                containers: vec![Container {
                    name: "trading-agent".to_string(),
                    image: Some(self.image_name.clone()),
                    image_pull_policy: Some("Always".to_string()), // Always pull latest for production
                    env: Some(env_vars),
                    resources: Some(ResourceRequirements {
                        limits: Some(resource_limits),
                        requests: Some(resource_requests),
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                restart_policy: Some("Always".to_string()),
                termination_grace_period_seconds: Some(30),
                ..Default::default()
            }),
            ..Default::default()
        };

        // Create Pod
        pods.create(&PostParams::default(), &pod)
            .await
            .context(format!("Failed to create pod {}", pod_name))?;

        tracing::info!("Successfully spawned agent pod {} for user {}", pod_name, user.id);
        Ok(pod_name)
    }

    pub async fn stop_agent(&self, user_id: &uuid::Uuid) -> Result<()> {
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);
        let pod_name = format!("agent-user-{}", user_id);

        pods.delete(&pod_name, &DeleteParams::default())
            .await
            .context(format!("Failed to delete pod {}", pod_name))?;

        tracing::info!("Successfully stopped agent pod {} for user {}", pod_name, user_id);
        Ok(())
    }

    pub async fn get_agent_status(&self, user_id: &uuid::Uuid) -> Result<Option<String>> {
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);
        let pod_name = format!("agent-user-{}", user_id);

        match pods.get(&pod_name).await {
            Ok(pod) => {
                let status = pod.status
                    .and_then(|s| s.phase)
                    .unwrap_or_else(|| "Unknown".to_string());
                Ok(Some(status))
            }
            Err(_) => Ok(None), // Pod doesn't exist
        }
    }

    pub async fn list_agent_pods(&self) -> Result<Vec<(uuid::Uuid, String, String)>> {
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);
        
        let list_params = kube::api::ListParams::default()
            .labels("app=scalping-agent");
            
        let pods_list = pods.list(&list_params).await?;
        
        let mut results = Vec::new();
        for pod in pods_list.items {
            if let Some(labels) = pod.metadata.labels {
                if let Some(user_id_str) = labels.get("user_id") {
                    if let Ok(user_id) = uuid::Uuid::parse_str(user_id_str) {
                        let pod_name = pod.metadata.name.unwrap_or_else(|| "Unknown".to_string());
                        let status = pod.status
                            .and_then(|s| s.phase)
                            .unwrap_or_else(|| "Unknown".to_string());
                        results.push((user_id, pod_name, status));
                    }
                }
            }
        }
        
        Ok(results)
    }
}
