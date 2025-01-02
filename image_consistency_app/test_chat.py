import requests
import json
from typing import Dict, Optional
import time

class AICharacterTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/chat"
        
    def send_message(self, message: str, context: str = "") -> Dict:
        """Send a message to the chat endpoint and return the response"""
        payload = {
            "user_input": message,
            "context": context
        }
        
        try:
            response = requests.post(self.chat_endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error sending message: {str(e)}")
            return {}

    def run_test_scenario(self, scenario_name: str, messages: list):
        """Run a test scenario with multiple messages"""
        print(f"\n=== Running Test Scenario: {scenario_name} ===")
        
        for i, message in enumerate(messages, 1):
            print(f"\nStep {i}: Sending message: '{message}'")
            response = self.send_message(message)
            
            print("Response:")
            print(f"- Text: {response.get('response', 'No text response')}")
            print(f"- Image Generated: {'Yes' if response.get('image_url') else 'No'}")
            if response.get('debug_info'):
                print(f"- Debug Info: {response['debug_info']}")
            
            # Add slight delay between messages
            time.sleep(1)

def main():
    # Initialize tester
    tester = AICharacterTester()
    
    # Test Scenario 1: Basic Conversation
    basic_conversation = [
        "Hi Rancho! How are you?",
        "Tell me about your college days",
        "What's your favorite invention?"
    ]
    tester.run_test_scenario("Basic Conversation", basic_conversation)
    
    # Test Scenario 2: Image Generation Triggers
    image_triggers = [
        "Can you show me what your college looks like?",
        "Now draw yourself standing in front of that college",
        "Can you show me your latest invention?"
    ]
    tester.run_test_scenario("Image Generation", image_triggers)
    
    # Test Scenario 3: Emotional Response
    emotional_scenario = [
        "What do you think about the education system?",
        "Can you show me how you envision the perfect classroom?",
        "Draw your ideal learning environment"
    ]
    tester.run_test_scenario("Emotional Response", emotional_scenario)
    
    # Test Scenario 4: Memory and Consistency
    consistency_test = [
        "Show me your dormitory room at the college",
        "What do you like about your room?",
        "Now show me your study corner in that same room",
        "Draw yourself studying with your friends in this room"
    ]
    tester.run_test_scenario("Memory and Consistency", consistency_test)
    
    # Test Scenario 5: Edge Cases
    edge_cases = [
        "",  # Empty message
        "Show me something" * 50,  # Very long message
        "!@#$%^&*()",  # Special characters
        "Generate an image",  # Direct command
    ]
    tester.run_test_scenario("Edge Cases", edge_cases)

if __name__ == "__main__":
    main()