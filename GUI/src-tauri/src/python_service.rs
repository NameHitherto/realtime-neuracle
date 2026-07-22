use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

pub struct PythonService {
    child: Arc<Mutex<Option<Child>>>,
    game_child: Arc<Mutex<Option<Child>>>,
    pub api_url: String,
    pub ws_url: String,
}

impl PythonService {
    pub fn new() -> Self {
        Self {
            child: Arc::new(Mutex::new(None)),
            game_child: Arc::new(Mutex::new(None)),
            api_url: "http://127.0.0.1:8000".to_string(),
            ws_url: "ws://127.0.0.1:8000/ws/telemetry".to_string(),
        }
    }

    pub async fn start(&self, python_path: &str, script_path: &str, args: Vec<String>) -> Result<(), String> {
        let mut child_guard = self.child.lock().await;
        if let Some(child) = child_guard.as_mut() {
            if child.try_wait().map_err(|e| e.to_string())?.is_none() {
                return Ok(());
            }
            *child_guard = None;
        }

        let resolved_script = resolve_script_path(script_path);
        let mut cmd = Command::new(python_path);
        cmd.arg(&resolved_script);
        for arg in args {
            cmd.arg(arg);
        }
        let child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start Python service: {}", e))?;

        *child_guard = Some(child);
        drop(child_guard);

        // Wait for service to be ready
        let client = reqwest::Client::new();
        for _ in 0..30 {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if let Ok(resp) = client.get(format!("{}/api/health", self.api_url)).send().await {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
        }

        self.stop().await;
        Err(format!(
            "Python service failed to start within timeout (script: {})",
            resolved_script.display()
        ))
    }

    pub async fn stop(&self) {
        let mut child_guard = self.child.lock().await;
        if let Some(ref mut child) = *child_guard {
            let _ = child.kill();
            let _ = child.wait();
        }
        *child_guard = None;
    }

    pub async fn start_game(&self, game_path: &str) -> Result<(), String> {
        let mut game_guard = self.game_child.lock().await;
        if game_guard.is_some() {
            return Ok(());
        }

        let child = Command::new(game_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start game: {}", e))?;

        *game_guard = Some(child);
        Ok(())
    }

    pub async fn stop_game(&self) {
        let mut game_guard = self.game_child.lock().await;
        if let Some(ref mut child) = *game_guard {
            let _ = child.kill();
            let _ = child.wait();
        }
        *game_guard = None;
    }
}

fn resolve_script_path(script_path: &str) -> PathBuf {
    let path = Path::new(script_path);
    if path.is_absolute() || path.exists() {
        return path.to_path_buf();
    }

    let project_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let normalized = script_path.trim_start_matches("../").trim_start_matches("..\\");
    let project_path = project_root.join(normalized);
    if project_path.exists() {
        project_path
    } else {
        path.to_path_buf()
    }
}

impl Default for PythonService {
    fn default() -> Self {
        Self::new()
    }
}
