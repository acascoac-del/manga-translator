import asyncio
import httpx
import base64
import os

async def test_modal_large():
    # Create a 3MB dummy string
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=" * 30000
    
    payload = {
        "image": b64,
        "target_lang": "es",
        "translator": "google",
        "detector": "ctd",
        "ocr": "manga_ocr",
        "inpainter": "lama",
        "renderer": "manga2eng"
    }

    url = "https://acasco-ac--main-fastapi-endpoint.modal.run/translate/json"
    
    try:
        async with httpx.AsyncClient(verify=False) as client:
            print(f"Sending request to {url} with size {len(b64)} bytes")
            resp = await client.post(url, json=payload, timeout=30.0)
            print("Status:", resp.status_code)
            print("Response:", resp.text[:200])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_modal_large())
