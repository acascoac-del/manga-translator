import asyncio
import httpx

async def test():
    async with httpx.AsyncClient(verify=False, timeout=10) as c:
        try:
            r = await c.get('https://lingva.pussthecat.org/api/v1/ja/es/hello')
            print(f"Status: {r.status_code}")
            print(f"Response: {r.text}")
        except Exception as e:
            print(e)

asyncio.run(test())
