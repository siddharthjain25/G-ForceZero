# Use a lightweight Ubuntu image
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies: g++, cmake, python3, pip, git, wget
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire G-ForceZero repository into the container
COPY . /app/G-ForceZero

# Compile the C++ Engine (using high performance flags)
WORKDIR /app/G-ForceZero/cpp_engine
RUN g++ -O3 -march=native nnue_engine.cpp nnue.cpp -o nnue_engine

# Clone the official Lichess Bot
WORKDIR /app
RUN git clone https://github.com/lichess-bot/lichess-bot.git

# Install Lichess Bot dependencies
WORKDIR /app/lichess-bot
RUN pip3 install -r requirements.txt
# Install Flask for our dummy web server to keep Render happy
RUN pip3 install Flask

# Copy our custom Render setup files into the bot directory
COPY G-ForceZero/render/config.yml /app/lichess-bot/config.yml
COPY G-ForceZero/render/keep_alive.py /app/lichess-bot/keep_alive.py
COPY G-ForceZero/render/start.sh /app/lichess-bot/start.sh

# Make start script executable
RUN chmod +x /app/lichess-bot/start.sh

# Command to run when the container starts
CMD ["./start.sh"]
