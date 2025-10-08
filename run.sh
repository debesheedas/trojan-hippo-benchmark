#!/bin/bash

# Email Agent MVP - Quick Start Script

echo "🚀 Starting Email Agent MVP..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from example..."
    echo "OPENAI_API_KEY=your-api-key-here" > .env
    echo "⚠️  Please edit .env and add your OpenAI API key if you want to use GPT models"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting server at http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

