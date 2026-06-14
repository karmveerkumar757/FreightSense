import os
import sys

print("Loading modules...")
try:
    from src.api.main import extract_and_advise
    from src.genai.agentic_rag import FallbackAdvisoryAgent
    from src.genai.rag_retriever import get_chroma_collection
    import asyncio
    print("Modules loaded.")
except Exception as e:
    print(f"Error loading modules: {e}")
    sys.exit(1)

async def test():
    print("Testing get_chroma_collection...")
    try:
        col = get_chroma_collection()
        print("Collection retrieved successfully.")
    except Exception as e:
        print(f"Error getting collection: {e}")

    print("Testing LLM Agent setup...")
    try:
        agent = FallbackAdvisoryAgent()
        print("Agent loaded successfully.")
    except Exception as e:
        print(f"Error loading agent: {e}")
        
    print("Testing extraction...")
    try:
        # Assuming extract_and_advise expects a string input directly for testing
        # wait, extract_and_advise is a FastAPI endpoint taking Form(...) and background tasks.
        pass
    except Exception as e:
        print(f"Error in extraction: {e}")

if __name__ == "__main__":
    asyncio.run(test())
