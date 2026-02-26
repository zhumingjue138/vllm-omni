#!/bin/bash
#
# L5 GPU 监控实时仪表板服务
# 读取 moniter.sh 写入的 current_run_id 与 latest.json，提供 Web 查看。
#
# 用法：./serve_dashboard.sh [端口]
#   端口默认 8765；需先启动 ./moniter.sh 才有数据。
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${GPU_MONITOR_DATA_ROOT:-$SCRIPT_DIR/gpu_monitor_data}"
PORT="${1:-8765}"

if ! command -v python3 &>/dev/null; then
    echo "错误：未找到 python3，无法启动仪表板服务。"
    exit 1
fi

cd "$SCRIPT_DIR"
echo "========================================"
echo "GPU 监控仪表板"
echo "数据目录: $DATA_ROOT"
echo "访问地址: http://127.0.0.1:$PORT/gpu_dashboard.html"
echo "按 Ctrl+C 停止服务"
echo "========================================"
exec python3 - "$PORT" "$DATA_ROOT" "$SCRIPT_DIR" << 'PY'
import http.server
import json
import os
import sys

PORT = int(sys.argv[1])
DATA_ROOT = sys.argv[2]
SCRIPT_DIR = sys.argv[3]

def get_latest_path():
    rid_file = os.path.join(DATA_ROOT, "current_run_id")
    if not os.path.isfile(rid_file):
        return None
    with open(rid_file, "r") as f:
        run_id = f.read().strip()
    path = os.path.join(DATA_ROOT, run_id, "latest.json")
    return path if os.path.isfile(path) else None

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/latest" or self.path == "/api/latest.json":
            path = get_latest_path()
            if not path:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"no current run or latest.json"}')
                return
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except Exception:
                self.send_response(500)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def log_message(self, format, *args):
        pass

with http.server.HTTPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
PY
