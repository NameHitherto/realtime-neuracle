mod commands;
mod python_service;
mod telemetry_client;

use std::sync::Arc;

use crate::commands::*;
use crate::python_service::PythonService;
use crate::telemetry_client::TelemetryClient;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Arc::new(PythonService::new()))
        .setup(|app| {
            let app_handle = app.handle().clone();

            // Start telemetry WebSocket client in background
            tauri::async_runtime::spawn(async move {
                // Start telemetry WebSocket client
                let client = TelemetryClient::new(
                    app_handle.clone(),
                    "ws://127.0.0.1:8000/ws/telemetry".to_string(),
                );
                client.start().await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_health,
            get_status,
            get_events,
            get_history,
            send_manual_control,
            get_default_launch_paths,
            start_python_service,
            check_python_exists,
            stop_python_service,
            start_game_process,
            stop_game_process,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
