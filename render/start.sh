#!/bin/bash

# 1. Inject the secure Render Environment Variable into the config.yml
if [ -n "$LICHESS_BOT_TOKEN" ]; then
    echo "Injecting Lichess API Token from Environment Variable..."
    sed -i "s/YOUR_API_TOKEN/$LICHESS_BOT_TOKEN/g" config.yml
else
    echo "WARNING: LICHESS_BOT_TOKEN environment variable is not set."
fi

# 2. Start the dummy web server in the background (Required by Render so the port binds)
echo "Starting Dummy Web Server..."
python3 keep_alive.py &

# 3. Start the actual Lichess bot in the foreground
echo "Starting G-ForceZero Lichess Bot..."
python3 lichess-bot.py
