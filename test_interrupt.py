#!/usr/bin/env python3
"""
Test script to verify KeyboardInterrupt handling in Excel AI Agent
"""
import subprocess
import signal
import time
import sys

def test_keyboard_interrupt():
    """Test KeyboardInterrupt handling"""
    print("Testing KeyboardInterrupt handling...")
    
    # Start the agent process
    process = subprocess.Popen(
        [sys.executable, 'excel_ai_agent.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give it a moment to start
    time.sleep(1)
    
    # Send SIGINT (equivalent to Ctrl+C)
    print("Sending SIGINT (Ctrl+C) to agent...")
    process.send_signal(signal.SIGINT)
    
    # Wait for the process to finish
    stdout, stderr = process.communicate(timeout=5)
    
    print("Agent output:")
    print(stdout)
    
    if "👋 Goodbye! Excel Agent session ended." in stdout:
        print("✅ SUCCESS: KeyboardInterrupt handled correctly!")
        return True
    else:
        print("❌ FAILURE: KeyboardInterrupt not handled properly")
        print("stderr:", stderr)
        return False

if __name__ == "__main__":
    success = test_keyboard_interrupt()
    sys.exit(0 if success else 1)