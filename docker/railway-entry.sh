#!/bin/sh
# docker/railway-entry.sh — Railway 啟動腳本
# railway.json 的 startCommand = "sh /opt/hermes/docker/railway-entry.sh"。
#
# 為什麼需要它：Railway 把持久 Volume 掛在 /opt/data 後，預設是 root 擁有，
# 而 hermes 跑在受限 user(UID 10000)。Railway 啟動時繞過了 image 內建的
# stage2-hook（負責 chown Volume），導致 gateway 要建 /opt/data/logs 時 EACCES 崩。
# 解法：本腳本若以 root 起，先把 Volume 擁有權交給 hermes，再 seed config，
# 最後 exec gateway（hermes gateway run 內部自會 drop 到 hermes user）。
# 開頭印診斷（uid + Volume 擁有權）方便確證。

HERMES_HOME="${HERMES_HOME:-/opt/data}"

echo "[railway-entry] uid=$(id -u) gid=$(id -g) HERMES_HOME=$HERMES_HOME"
echo "[railway-entry] before: $(ls -ld "$HERMES_HOME" 2>&1)"

mkdir -p "$HERMES_HOME" 2>/dev/null || true

# 若以 root 起：把整個 Volume 交給 hermes，gateway drop 後才寫得進去
if [ "$(id -u)" = "0" ]; then
  chown -R hermes:hermes "$HERMES_HOME" 2>&1 | head -1 || true
  echo "[railway-entry] chowned $HERMES_HOME -> hermes; after: $(ls -ld "$HERMES_HOME" 2>&1)"
fi

# 首次 seed 最小 config（model=Gemini 直連、LINE 開）；只在不存在時寫
if [ ! -f "$HERMES_HOME/config.yaml" ]; then
  cat > "$HERMES_HOME/config.yaml" <<'YAML'
model:
  default: gemini/gemini-2.5-flash
gateway:
  platforms:
    line:
      enabled: true
YAML
  [ "$(id -u)" = "0" ] && chown hermes:hermes "$HERMES_HOME/config.yaml" 2>/dev/null || true
  echo "[railway-entry] seeded config.yaml"
fi

exec hermes gateway run
