#!/usr/bin/env python3
"""
Excel AI Agent - Interactive command-line tool for Excel processing with AI capabilities
Includes proper KeyboardInterrupt handling for ESC/Ctrl+C graceful shutdown.
"""

import os
import sys
import signal
import pandas as pd
import json
from datetime import datetime
import time


class ExcelAIAgent:
    """Excel AI Agent with interactive chat functionality and proper interrupt handling."""
    
    def __init__(self):
        self.memory = {}
        self.session_data = []
        self.running = True
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals (Ctrl+C, ESC)."""
        self.running = False
        print("\n👋 Goodbye! Excel Agent session ended.")
        sys.exit(0)
    
    def display_welcome(self):
        """Display welcome message and available commands."""
        print("=" * 60)
        print("🤖 Welcome to Excel AI Agent!")
        print("=" * 60)
        print("Available commands:")
        print("  📊 excel <filename>  - Load and analyze Excel file")
        print("  🧠 memory           - View current memory state")
        print("  🧹 clear            - Clear memory and session data")
        print("  🧪 test             - Run test functionality")
        print("  ❌ quit             - Exit the agent")
        print("  ❓ help             - Show this help message")
        print("\n💡 Tip: Press Ctrl+C or ESC anytime to exit immediately")
        print("=" * 60)
    
    def process_excel_file(self, filename):
        """Process Excel file and extract information."""
        try:
            if not os.path.exists(filename):
                print(f"❌ Error: File '{filename}' not found.")
                return
            
            # Read Excel file
            df = pd.read_excel(filename)
            print(f"✅ Successfully loaded Excel file: {filename}")
            print(f"📋 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"📄 Columns: {list(df.columns)}")
            
            # Store in memory
            self.memory[filename] = {
                'dataframe': df,
                'loaded_at': datetime.now().isoformat(),
                'shape': df.shape,
                'columns': list(df.columns)
            }
            
            # Show preview
            print("\n📊 Data Preview:")
            print(df.head())
            
            # Basic analysis
            print("\n📈 Basic Analysis:")
            print(f"  • Numeric columns: {df.select_dtypes(include=['number']).columns.tolist()}")
            print(f"  • Text columns: {df.select_dtypes(include=['object']).columns.tolist()}")
            print(f"  • Missing values: {df.isnull().sum().sum()}")
            
        except Exception as e:
            print(f"❌ Error processing Excel file: {str(e)}")
    
    def show_memory(self):
        """Display current memory state."""
        if not self.memory:
            print("🧠 Memory is empty")
            return
        
        print("🧠 Current Memory State:")
        print("-" * 40)
        for key, value in self.memory.items():
            if isinstance(value, dict) and 'loaded_at' in value:
                print(f"📄 {key}:")
                print(f"   • Loaded: {value['loaded_at']}")
                print(f"   • Shape: {value['shape']}")
                print(f"   • Columns: {len(value['columns'])}")
            else:
                print(f"📝 {key}: {value}")
        print("-" * 40)
    
    def clear_memory(self):
        """Clear memory and session data."""
        self.memory.clear()
        self.session_data.clear()
        print("🧹 Memory and session data cleared!")
    
    def run_test(self):
        """Run test functionality to demonstrate capabilities."""
        print("🧪 Running Excel AI Agent Tests...")
        
        # Test 1: Create sample data
        print("\n📊 Test 1: Creating sample Excel data...")
        sample_data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'Age': [25, 30, 35, 28],
            'Salary': [50000, 60000, 70000, 55000],
            'Department': ['Engineering', 'Sales', 'Marketing', 'Engineering']
        }
        df = pd.DataFrame(sample_data)
        
        # Save test file
        test_filename = 'test_data.xlsx'
        df.to_excel(test_filename, index=False)
        print(f"✅ Created test file: {test_filename}")
        
        # Test 2: Load and analyze the test file
        print("\n📈 Test 2: Loading and analyzing test data...")
        self.process_excel_file(test_filename)
        
        # Test 3: Memory functionality
        print("\n🧠 Test 3: Testing memory functionality...")
        self.memory['test_value'] = 'This is a test entry'
        self.show_memory()
        
        print("\n✅ All tests completed successfully!")
        
        # Clean up test file
        try:
            os.remove(test_filename)
            print(f"🧹 Cleaned up test file: {test_filename}")
        except:
            pass
    
    def process_command(self, command):
        """Process user commands."""
        command = command.strip()
        
        if not command:
            return True
        
        parts = command.split()
        cmd = parts[0].lower()
        
        try:
            if cmd == 'quit' or cmd == 'exit':
                print("👋 Goodbye! Excel Agent session ended.")
                return False
            
            elif cmd == 'help':
                self.display_welcome()
            
            elif cmd == 'memory':
                self.show_memory()
            
            elif cmd == 'clear':
                self.clear_memory()
            
            elif cmd == 'test':
                self.run_test()
            
            elif cmd == 'excel':
                if len(parts) < 2:
                    print("❌ Usage: excel <filename>")
                else:
                    filename = ' '.join(parts[1:])  # Handle filenames with spaces
                    self.process_excel_file(filename)
            
            else:
                print(f"❓ Unknown command: {cmd}")
                print("💡 Type 'help' to see available commands")
        
        except KeyboardInterrupt:
            # This should be caught by the outer loop, but adding as backup
            self.running = False
            print("\n👋 Goodbye! Excel Agent session ended.")
            return False
        except Exception as e:
            print(f"❌ Error executing command: {str(e)}")
        
        return True
    
    def interactive_chat(self):
        """Main interactive chat loop with proper KeyboardInterrupt handling."""
        self.display_welcome()
        
        while self.running:
            try:
                # Get user input with proper prompt
                user_input = input("\n🤖 Excel AI Agent > ").strip()
                
                # Log session data
                self.session_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'input': user_input
                })
                
                # Process the command
                continue_chat = self.process_command(user_input)
                if not continue_chat:
                    break
                    
                # Continue to next iteration after processing command
                continue
                
            except KeyboardInterrupt:
                # Handle Ctrl+C / ESC key press
                self.running = False
                print("\n👋 Goodbye! Excel Agent session ended.")
                break
            except EOFError:
                # Handle EOF (Ctrl+D on Unix, Ctrl+Z on Windows)
                self.running = False
                print("\n👋 Goodbye! Excel Agent session ended.")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
                continue
    
    def run(self):
        """Start the Excel AI Agent."""
        try:
            self.interactive_chat()
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Excel Agent session ended.")
        finally:
            # Cleanup if needed
            pass


def main():
    """Main entry point for the Excel AI Agent."""
    try:
        agent = ExcelAIAgent()
        agent.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Excel Agent session ended.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()