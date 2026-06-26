#!/command/with-contenv sh
# docker/railway-entry.sh — Railway 啟動腳本
# main-wrapper.sh 會把第一個 arg 當 executable 直接 exec，所以
# railway.json 的 startCommand 用 "sh /opt/hermes/docker/railway-entry.sh"。
# 作用：首次開機若 Volume(/opt/data) 沒有 config.yaml，就 seed 一份最小設定
# （model=Gemini 直連、LINE 平台開啟），再 exec 正式的 gateway。
# 只在「不存在時」寫 → 日後手改 config 或第二步加 Honcho 都不會被覆蓋。
set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
mkdir -p "$HERMES_HOME"

if [ ! -f "$HERMES_HOME/config.yaml" ]; then
  cat > "$HERMES_HOME/config.yaml" <<'YAML'
model:
  default: gemini/gemini-2.5-flash
gateway:
  platforms:
    line:
      enabled: true
YAML
fi

exec hermes gateway run
