import asyncio
from app.engine.translation import groq_translate

async def test():
    # Will fail without a key
    print("Testing translation with bad key...")
    res = await groq_translate.translate_text("こんにちは", api_key="bad_key", model="llama-3.1-8b-instant")
    print(f"Result: {res}")

asyncio.run(test())
