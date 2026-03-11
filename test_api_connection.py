"""
Quick API connection test - verifies LLM is callable before running full generation.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

def test_llm_api():
    print("\n" + "="*80)
    print("LLM API CONNECTION TEST")
    print("="*80 + "\n")
    
    # Load credentials
    load_dotenv(".env")
    
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")
    
    if not api_key:
        print("✗ Error: LLM_API_KEY not found in .env file")
        print("\nPlease create .env file with:")
        print("  LLM_API_KEY=your-api-key")
        print("  LLM_BASE_URL=your-api-url")
        print("  LLM_MODEL_NAME=your-model")
        return False
    
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {api_key[:8]}...{api_key[-4:]}\n")
    
    # Test connection
    print("Testing API connection with simple request...")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Connection successful!'"}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        
        print(f"\n✓ SUCCESS!")
        print(f"  Response: {result}")
        print(f"  Model used: {response.model}")
        print(f"  Tokens: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
        
        print("\n" + "="*80)
        print("API is working! You can now run:")
        print("  python test_synthetic_generation.py")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED!")
        print(f"  Error: {str(e)}\n")
        
        print("Common issues:")
        print("  1. Invalid API key")
        print("  2. Wrong base URL")
        print("  3. Model not available/incorrect name")
        print("  4. Network/firewall issues")
        print("  5. API rate limits\n")
        
        return False

if __name__ == "__main__":
    test_llm_api()
