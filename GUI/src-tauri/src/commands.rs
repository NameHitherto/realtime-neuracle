use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::path::{Path, PathBuf};
use tauri::State;

use crate::python_service::PythonService;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Telemetry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ts: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eeg: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lsl: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub game: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recording: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Event {
    pub ts: f64,
    pub level: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Health {
    pub ok: bool,
    pub telemetry: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StartServicePayload {
    pub python_path: String,
    pub script_path: String,
    pub args: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StartGamePayload {
    pub game_path: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DefaultLaunchPaths {
    pub python_path: String,
    pub script_path: String,
    pub game_path: String,
}

pub(crate) fn project_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .map_err(|e| format!("Failed to resolve project root: {}", e))
}

pub(crate) fn resolve_project_path(value: &str) -> Result<PathBuf, String> {
    let path = Path::new(value.trim());
    if path.as_os_str().is_empty() {
        return Err("Path cannot be empty".to_string());
    }
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(project_root()?.join(path))
    }
}

#[cfg(test)]
mod path_tests {
    use super::{get_default_launch_paths, resolve_project_path};
    use std::path::Path;

    #[test]
    fn default_launch_paths_are_relative_and_resolve_to_files() {
        let defaults = get_default_launch_paths().expect("default paths");
        for value in [
            defaults.python_path,
            defaults.script_path,
        ] {
            assert!(!Path::new(&value).is_absolute(), "{value} should be relative");
            assert!(
                resolve_project_path(&value).expect("resolved path").is_file(),
                "{value} should resolve to an existing file"
            );
        }
    }
}

#[tauri::command]
pub fn get_default_launch_paths() -> Result<DefaultLaunchPaths, String> {
    Ok(DefaultLaunchPaths {
        python_path: r".venv\Scripts\python.exe".to_string(),
        script_path: "dashboard_service.py".to_string(),
        // The race client normally runs on another PC. Local launch is optional.
        game_path: std::env::var("BCI_GAME_EXE").unwrap_or_else(|_| {
            let local = project_root().unwrap_or_default()
                .join(r"..\虚拟任务竞速赛_0911_禁键盘_比赛版本\虚拟任务竞速赛.exe");
            if local.is_file() { local.to_string_lossy().into_owned() } else { String::new() }
        }),
    })
}

#[tauri::command]
pub async fn get_health(service: State<'_, Arc<PythonService>>) -> Result<Health, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/api/health", service.api_url))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    if !resp.status().is_success() {
        return Err(format!("HTTP error: {}", resp.status()));
    }
    
    let data: Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(Health {
        ok: data.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
        telemetry: data.get("telemetry").cloned(),
    })
}

#[tauri::command]
pub async fn get_status(service: State<'_, Arc<PythonService>>) -> Result<Telemetry, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/api/status", service.api_url))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    if !resp.status().is_success() {
        return Err(format!("HTTP error: {}", resp.status()));
    }
    
    let data: Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(value_to_telemetry(data))
}

#[tauri::command]
pub async fn get_events(service: State<'_, Arc<PythonService>>, limit: u32) -> Result<Vec<Event>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/api/events?limit={}", service.api_url, limit))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    if !resp.status().is_success() {
        return Err(format!("HTTP error: {}", resp.status()));
    }
    
    let data: Vec<Value> = resp.json().await.map_err(|e| e.to_string())?;
    let events: Vec<Event> = data.into_iter().map(|v| Event {
        ts: v.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0),
        level: v.get("level").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        message: v.get("message").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        details: v.get("details").cloned(),
    }).collect();
    Ok(events)
}

#[tauri::command]
pub async fn get_history(service: State<'_, Arc<PythonService>>, limit: u32) -> Result<Vec<Telemetry>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/api/history?limit={}", service.api_url, limit))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    if !resp.status().is_success() {
        return Err(format!("HTTP error: {}", resp.status()));
    }
    
    let data: Vec<Value> = resp.json().await.map_err(|e| e.to_string())?;
    Ok(data.into_iter().map(value_to_telemetry).collect())
}

#[tauri::command]
pub async fn send_manual_control(
    service: State<'_, Arc<PythonService>>,
    control: u8,
    duration_ms: u32,
) -> Result<Telemetry, String> {
    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "control": control,
        "duration_ms": duration_ms,
    });
    
    let resp = client
        .post(format!("{}/api/test/control", service.api_url))
        .json(&payload)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    if !resp.status().is_success() {
        return Err(format!("HTTP error: {}", resp.status()));
    }
    
    let data: Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(value_to_telemetry(data))
}

#[tauri::command]
pub async fn start_python_service(
    service: State<'_, Arc<PythonService>>,
    payload: StartServicePayload,
) -> Result<(), String> {
    service.start(&payload.python_path, &payload.script_path, payload.args).await
}

#[tauri::command]
pub async fn check_python_exists(path: String) -> Result<bool, String> {
    let exists = resolve_project_path(&path)?.is_file();
    Ok(exists)
}

#[tauri::command]
pub async fn stop_python_service(service: State<'_, Arc<PythonService>>) -> Result<(), String> {
    service.stop().await;
    Ok(())
}

#[tauri::command]
pub async fn start_game_process(
    service: State<'_, Arc<PythonService>>,
    payload: StartGamePayload,
) -> Result<(), String> {
    service.start_game(&payload.game_path).await
}

#[tauri::command]
pub async fn stop_game_process(service: State<'_, Arc<PythonService>>) -> Result<(), String> {
    service.stop_game().await;
    Ok(())
}

fn value_to_telemetry(value: Value) -> Telemetry {
    Telemetry {
        seq: value.get("seq").and_then(|v| v.as_u64()),
        ts: value.get("ts").and_then(|v| v.as_f64()),
        status: value.get("status").cloned(),
        health: value.get("health").cloned(),
        eeg: value.get("eeg").cloned(),
        model: value.get("model").cloned(),
        lsl: value.get("lsl").cloned(),
        game: value.get("game").cloned(),
        recording: value.get("recording").cloned(),
        calibration: value.get("calibration").cloned(),
    }
}
