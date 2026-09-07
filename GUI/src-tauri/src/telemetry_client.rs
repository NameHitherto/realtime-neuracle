use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::time::sleep;
use futures_util::StreamExt;

pub struct TelemetryClient {
    app: AppHandle,
    ws_url: String,
    running: Arc<tokio::sync::RwLock<bool>>,
}

impl TelemetryClient {
    pub fn new(app: AppHandle, ws_url: String) -> Self {
        Self {
            app,
            ws_url,
            running: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }

    pub async fn start(&self) {
        let mut running = self.running.write().await;
        if *running {
            return;
        }
        *running = true;
        drop(running);

        let app = self.app.clone();
        let ws_url = self.ws_url.clone();
        let running_flag = self.running.clone();

        tokio::spawn(async move {
            let mut retry_delay = Duration::from_secs(1);
            let max_retry_delay = Duration::from_secs(30);

            loop {
                if !*running_flag.read().await {
                    break;
                }

                match tokio_tungstenite::connect_async(&ws_url).await {
                    Ok((mut ws_stream, _)) => {
                        retry_delay = Duration::from_secs(1);
                        let _ = app.emit("telemetry-connected", true);

                        while let Some(msg) = ws_stream.next().await {
                            if !*running_flag.read().await {
                                break;
                            }
                            match msg {
                                Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                                    if let Ok(payload) = serde_json::from_str::<Value>(&text) {
                                        let _ = app.emit("telemetry-update", payload);
                                    }
                                }
                                Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => break,
                                Err(_) => break,
                                _ => {}
                            }
                        }

                        let _ = app.emit("telemetry-connected", false);
                    }
                    Err(e) => {
                        log::warn!("WebSocket connection failed: {}", e);
                        let _ = app.emit("telemetry-connected", false);
                    }
                }

                sleep(retry_delay).await;
                retry_delay = std::cmp::min(retry_delay * 2, max_retry_delay);
            }
        });
    }
}
