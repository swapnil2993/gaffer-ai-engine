#!/usr/bin/env bash

# Configuration
DB_LOCK_FILE="scoutintel_local.db/LOCK"
PORT=8000
FRONTEND_DIR="frontend"
FRONTEND_PORT=3000

echo "🚀 Starting Gaffer AI Engine (backend + frontend)..."

# 1. Ensure no previous instance is running and clear the Milvus lock.
# Milvus Lite allows only ONE process to open the DB, so a leftover backend
# (or one started in another terminal) blocks startup. Kill prior instances and
# anything holding the lock file.
echo "🔍 Stopping any previous instance / lock holders..."
pkill -f "uvicorn backend.app.main" 2>/dev/null

# Free the backend AND frontend ports — a leftover uvicorn or react-scripts
# (often from a previous run that didn't shut down cleanly) blocks startup
# ("Something is already running on port 3000").
for p in "$PORT" "$FRONTEND_PORT"; do
    PORT_PIDS=$(lsof -ti "tcp:$p" 2>/dev/null | sort -u)
    if [ -n "$PORT_PIDS" ]; then
        echo "⚠️  Freeing port $p (killing: $(echo $PORT_PIDS | tr '\n' ' '))"
        for pid in $PORT_PIDS; do kill -9 "$pid" 2>/dev/null; done
    fi
done

LOCKING_PIDS=$(lsof -t "$DB_LOCK_FILE" 2>/dev/null; lsof -t +D "scoutintel_local.db" 2>/dev/null)
LOCKING_PIDS=$(echo "$LOCKING_PIDS" | sort -u | tr '\n' ' ')
if [ -n "${LOCKING_PIDS// /}" ]; then
    echo "⚠️  Terminating lock holders: $LOCKING_PIDS"
    for pid in $LOCKING_PIDS; do
        kill -9 "$pid" 2>/dev/null
    done
fi
sleep 2

# Force remove the lock file just in case it's stale on disk
if [ -f "$DB_LOCK_FILE" ]; then
    echo "🧹 Removing stale lock file: $DB_LOCK_FILE"
    rm -f "$DB_LOCK_FILE"
fi

# 2. Verify .env existence
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found. Please create one with OPENAI_API_KEY or OPENROUTER_API_KEY."
    exit 1
fi

# 3. Load environment variables
echo "✅ Environment loaded."
set -a
source .env
set +a

# 4. Ensure frontend dependencies are installed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
fi

# Track child PIDs so we can shut both down cleanly on exit
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "🛑 Shutting down Gaffer AI Engine..."
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null
    # uvicorn --reload spawns a worker, which in turn spawns Milvus Lite's
    # embedded server as a multiprocessing child (its cmdline does NOT contain
    # "uvicorn", so pkill misses it). Kill by the DB lock file to release it.
    pkill -f "uvicorn backend.app.main" 2>/dev/null
    for pid in $(lsof -t "$DB_LOCK_FILE" 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
    # Frontend: npm start spawns a child node server that can outlive $FRONTEND_PID,
    # so also free the port directly.
    for pid in $(lsof -ti "tcp:$FRONTEND_PORT" 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
    wait 2>/dev/null
    exit "${1:-0}"
}
trap cleanup SIGINT SIGTERM

# 5. Start the backend (Uvicorn) in the background
echo "🔧 Starting backend on port $PORT..."
uv run uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT" --reload &
BACKEND_PID=$!

# 6. Wait for the backend to become healthy. Under --reload uvicorn keeps its
# supervisor alive even when startup FAILS, so we can't rely on process death —
# we poll the HTTP endpoint and bail (killing nothing but the backend) if it
# never comes up. The frontend is only started once the backend is serving.
echo "⏳ Waiting for backend to be ready on :$PORT..."
BACKEND_READY=0
for _ in $(seq 1 45); do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "❌ Backend process exited during startup."
        break
    fi
    if curl -sf "http://localhost:$PORT/" >/dev/null 2>&1; then
        BACKEND_READY=1
        break
    fi
    sleep 1
done

if [ "$BACKEND_READY" -ne 1 ]; then
    echo "❌ Backend failed to start — not launching the frontend."
    echo "   Check the logs above (common cause: Milvus lock held by another process)."
    cleanup 1
fi
echo "✅ Backend healthy: http://localhost:$PORT (API docs at /docs)"

# 7. Start the frontend (Create React App) in the background
echo "🎨 Starting frontend on port $FRONTEND_PORT..."
(cd "$FRONTEND_DIR" && PORT="$FRONTEND_PORT" BROWSER=none npm start) &
FRONTEND_PID=$!

echo ""
echo "✅ Frontend: http://localhost:$FRONTEND_PORT"
echo "Press Ctrl+C to stop both."

# Monitor: if EITHER process exits (e.g. backend crashes later), tear down both.
# (Polling with `kill -0` instead of `wait -n`, which macOS's bash 3.2 lacks.)
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
    sleep 1
done
echo "⚠️  A process exited — shutting everything down."
cleanup
