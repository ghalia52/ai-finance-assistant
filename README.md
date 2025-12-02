# 🤖 AI Financial Assistant (Finley)

An intelligent personal finance assistant powered by Google's Gemini AI. Track expenses, set budgets, and get AI-powered financial insights through a beautiful chat interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue.svg)

## ✨ Features

- 💬 **Natural Language Interface** - Chat with Finley using everyday language
- 💰 **Expense Tracking** - Automatically log and categorize expenses
- 📊 **Budget Management** - Set spending limits and get warnings
- 📈 **Spending Summaries** - Get detailed breakdowns by category
- 🤖 **AI-Powered** - Powered by Google Gemini 1.5 Pro
- 💾 **PostgreSQL Backend** - Robust data storage and retrieval
- 🎨 **Modern UI** - Beautiful React interface with glassmorphism design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL
- Node.js and npm
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/ai-finance-assistant.git
   cd ai-finance-assistant
```

2. **Set up environment variables**
```bash
   copy .env.example .env
```
   Edit `.env` and add your credentials

3. **Install Python dependencies**
```bash
   pip install -r requirements.txt
```

4. **Create PostgreSQL database**
```bash
   createdb finance_db
```

5. **Run the backend**
```bash
   python localGPT.py
```

6. **Install and run frontend** (in new terminal)
```bash
   cd frontend
   npm install
   npm start
```

## 📝 Usage

Type natural language commands:
- "What's my balance?"
- "I spent $45 on groceries"
- "Set budget for entertainment to $300"
- "Show spending summary"

## 🔧 Configuration

All settings are in `.env` file. See `.env.example` for available options.

## 📄 License

MIT License

## 🙏 Acknowledgments

- Google Gemini AI
- FastAPI
- React