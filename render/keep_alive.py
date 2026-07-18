from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "G-ForceZero Lichess Bot is ALIVE and running on Render! ♟️"

def run():
    # Render assigns a dynamic port using the PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    run()
