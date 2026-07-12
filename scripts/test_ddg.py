import asyncio
import httpx
import re
import ssl

async def test():
    # Bypass SSL for testing
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    async with httpx.AsyncClient(timeout=15.0, verify=context) as c:
        r = await c.get('https://duckduckgo.com/?q=translate', headers={'User-Agent':'Mozilla/5.0'})
        m = re.search(r'vqd=([\d-]+)', r.text)
        v = m.group(1) if m else '4-239123'
        print(f"VQD: {v}")
        
        r2 = await c.post('https://duckduckgo.com/translation.js', data={'vqd':v,'query':'hello','from':'en','to':'es'}, headers={'User-Agent':'Mozilla/5.0'})
        print(f"Status: {r2.status_code}")
        print(f"Response: {r2.text}")

asyncio.run(test())
