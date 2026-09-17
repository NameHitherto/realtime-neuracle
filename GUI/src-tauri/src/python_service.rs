use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::sync::Mutex;

use crate::commands::{project_root, resolve_project_path};

pub struct PythonService {
    child: Arc<Mutex<Option<Child>>>,
    game_child: Arc<Mutex<Option<Child>>>,
    stderr_tail: Arc<StdMutex<String>>,
    pub api_url: String,
}

impl PythonService {
    pub fn new() -> Self {
        Self {
            child: Arc::new(Mutex::new(None)),
            game_child: Arc::new(Mutex::new(None)),
            stderr_tail: Arc::new(StdMutex::new(String::new())),
            api_url: "http://127.0.0.1:8000".to_string(),
        }
    }

    pub async fn start(
        &self,
        python_path: &str,
        script_path: &str,
        args: Vec<String>,
    ) -> Result<(), String> {
        let mut child_guard = self.child.lock().await;
        if let Some(child) = child_guard.as_mut() {
            if child.try_wait().map_err(|e| e.to_string())?.is_none() {
                return Ok(());
            }
            *child_guard = None;
        }

        let resolved_python = resolve_project_path(python_path)?;
        let resolved_script = resolve_project_path(script_path)?;
        if !resolved_python.is_file() {
            return Err(format!("Python executable does not exist: {}", resolved_python.display()));
        }
        if !resolved_script.is_file() {
            return Err(format!("Python script does not exist: {}", resolved_script.display()));
        }
        let mut cmd = Command::new(&resolved_python);
        cmd.current_dir(project_root()?);
        cmd.arg(&resolved_script);
        for arg in args {
            cmd.arg(arg);
        }
        if let Ok(mut tail) = self.stderr_tail.lock() {
            tail.clear();
        }
        let mut child = cmd
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start Python service: {}", e))?;
        if let Some(stderr) = child.stderr.take() {
            let stderr_tail = self.stderr_tail.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    if let Ok(mut tail) = stderr_tail.lock() {
                        tail.push_str(&line);
                        tail.push('\n');
                        if tail.len() > 16_384 {
                            let mut split_at = tail.len() - 16_384;
                            while !tail.is_char_boundary(split_at) { split_at += 1; }
                            tail.drain(..split_at);
                        }
                    }
                }
            });
        }

        *child_guard = Some(child);
        drop(child_guard);

        // Wait for service to be ready
        let client = reqwest::Client::new();
        for _ in 0..30 {
            tokio::time::sleep(Duration::from_millis(500)).await;
            {
                let mut guard = self.child.lock().await;
                if let Some(child) = guard.as_mut() {
                    if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                        *guard = None;
                        let stderr = self
                            .stderr_tail
                            .lock()
                            .map(|tail| tail.clone())
                            .unwrap_or_default();
                        let detail = stderr.trim();
                        return Err(if detail.is_empty() {
                            format!("Python service exited early with {}", status)
                        } else {
                            format!("Python service exited early: {}", detail)
                        });
                    }
                }
            }
            if let Ok(resp) = client
                .get(format!("{}/api/health", self.api_url))
                .send()
                .await
            {
                if resp.status().is_success() {
                    if let Ok(body) = resp.json::<serde_json::Value>().await {
                        if body
                            .get("ok")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false)
                        {
                            return Ok(());
                        }
                        let inference = body
                            .pointer("/status/inference")
                            .and_then(|value| value.as_str());
                        if inference == Some("error") {
                            let message = body
                                .pointer("/telemetry/message")
                                .and_then(|value| value.as_str())
                                .unwrap_or("Python inference runtime failed");
                            self.stop().await;
                            return Err(message.to_string());
                        }
                    }
                }
            }
        }

        self.stop().await;
        Err(format!(
            "Python service failed to start within timeout (script: {})",
            resolved_script.display()
        ))
    }

    pub async fn stop_owned(&self) {
        let owns_child = self.child.lock().await.is_some();
        if owns_child { self.stop().await; }
    }

    pub async fn stop(&self) {
        // Ask Python to stop the acquisition loop and flush all experiment
        // files before terminating the process. A failed/unresponsive backend
        // still falls through to the hard process stop below.
        if let Ok(client) = reqwest::Client::builder()
            .timeout(Duration::from_secs(4))
            .build()
        {
            let _ = client
                .post(format!("{}/api/runtime/stop", self.api_url))
                .send()
                .await;
        }
        let mut child_guard = self.child.lock().await;
        if let Some(ref mut child) = *child_guard {
            let _ = child.kill();
            let _ = child.wait();
        }
        *child_guard = None;
    }

    pub async fn start_game(&self, game_path: &str) -> Result<(), String> {
        let mut game_guard = self.game_child.lock().await;
        if let Some(child) = game_guard.as_mut() {
            if child.try_wait().map_err(|e| e.to_string())?.is_none() {
                return Ok(());
            }
            *game_guard = None;
        }

        let resolved_game = resolve_project_path(game_path)?;
        if !resolved_game.is_file() {
            return Err(format!("Game executable does not exist: {}", resolved_game.display()));
        }
        let game_dir = resolved_game
            .parent()
            .ok_or_else(|| format!("Game directory is invalid: {}", resolved_game.display()))?;
        let child = Command::new(&resolved_game)
            .current_dir(game_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
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

impl Default for PythonService {
    fn default() -> Self {
        Self::new()
    }
}
