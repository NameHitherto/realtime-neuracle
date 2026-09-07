# BCI Racing Control Room

这是 YHC 实时脑电赛车系统的 React + TypeScript + Tauri 桌面界面。必须通过 Tauri 启动，直接打开 Vite 网页无法调用本地 Python/游戏进程命令。

从项目根目录启动：

```powershell
.\05_run_gui.ps1
```

或在本目录启动开发模式：

```powershell
npm install
npm run tauri dev
```

检查和构建：

```powershell
npm run lint
npm run build
cargo check --manifest-path .\src-tauri\Cargo.toml
```

GUI 固定使用 YHC 模型，并通过 Rust 后端启动项目 `.venv` 中的 Python 服务和相邻目录的 Unity 游戏。遥测由 Python FastAPI WebSocket 发送到 Rust，再作为 Tauri 事件推送给 React。
