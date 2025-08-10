#!/bin/bash

echo "🔧 Starting setup for Python 3.10 environment..."

# Check if Python 3.10 is available via apt (Deadsnakes method)
echo "📦 Attempting to install via Deadsnakes PPA..."
sudo apt update
sudo apt install -y software-properties-common

if sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update; then
    sudo apt install -y python3.10 python3.10-venv
    echo "✅ Python 3.10 installed via apt. Creating virtual environment..."
    python3.10 -m venv myenv
    echo "✅ Virtual environment 'myenv' created. Run 'source myenv/bin/activate' to activate it."
    exit 0
else
    echo "⚠️ Deadsnakes method failed. Falling back to pyenv..."
fi

# Install dependencies for building Python
echo "📦 Installing build dependencies for pyenv..."
sudo apt update && sudo apt install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# Install pyenv
if ! command -v pyenv &> /dev/null; then
    echo "📥 Installing pyenv..."
    curl https://pyenv.run | bash

    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

    source ~/.bashrc
else
    echo "✅ pyenv already installed."
fi

# Install Python 3.10 using pyenv
echo "📥 Installing Python 3.10 with pyenv..."
~/.pyenv/bin/pyenv install 3.10.13
~/.pyenv/bin/pyenv global 3.10.13

# Create venv
python3.10 -m venv myenv
echo "✅ Python 3.10 installed and virtual environment 'myenv' created."
echo "Run 'source myenv/bin/activate' to activate it."
