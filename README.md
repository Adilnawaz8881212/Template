# Excel AI Agent

Interactive command-line Excel processing tool with AI capabilities and proper KeyboardInterrupt handling.

## Features

- 🤖 Interactive chat interface
- 📊 Excel file loading and analysis
- 🧠 Memory management for session data
- 🧹 Clear memory and session data
- 🧪 Built-in test functionality
- ❌ Graceful exit with ESC/Ctrl+C handling
- ⚠️ Robust error handling

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Excel AI Agent:
```bash
python3 excel_ai_agent.py
```

## Available Commands

- `excel <filename>` - Load and analyze Excel file
- `memory` - View current memory state
- `clear` - Clear memory and session data
- `test` - Run test functionality
- `help` - Show help message
- `quit` - Exit the agent

## KeyboardInterrupt Handling

The agent properly handles ESC/Ctrl+C interrupts:
- Press Ctrl+C or ESC anytime to exit immediately
- Shows "👋 Goodbye! Excel Agent session ended." message
- Graceful shutdown without hanging

## Example Usage

```bash
$ python3 excel_ai_agent.py
============================================================
🤖 Welcome to Excel AI Agent!
============================================================
Available commands:
  📊 excel <filename>  - Load and analyze Excel file
  🧠 memory           - View current memory state
  🧹 clear            - Clear memory and session data
  🧪 test             - Run test functionality
  ❌ quit             - Exit the agent
  ❓ help             - Show this help message

💡 Tip: Press Ctrl+C or ESC anytime to exit immediately
============================================================

🤖 Excel AI Agent > test
🧪 Running Excel AI Agent Tests...
...
🤖 Excel AI Agent > quit
👋 Goodbye! Excel Agent session ended.
```

## Original Audio-to-PDF Processor

The repository also contains the original Streamlit audio-to-PDF processing application in `main.py`.