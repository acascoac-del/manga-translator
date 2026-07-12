import asyncio
import httpx
import base64
import json

async def test():
    # create a dummy base64 image (small)
    # 1x1 white pixel
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
    
    payload = {
        "image": b64,
        "target_lang": "es",
        "translator": "google",
        "ollama_model": "llama3",
        "ollama_host": "http://localhost:8080",
        "detector": "ctd",
        "ocr": "manga_ocr",
        "inpainter": "lama",
        "renderer": "manga2eng"
    }

    async with httpx.AsyncClient() as client:
        print("Sending request to /translate/json")
        resp = await client.post("http://127.0.0.1:8001/translate/json", json=payload, timeout=60.0)
        print("Status:", resp.status_code)
        print("Response:", resp.text)

if __name__ == "__main__":
    asyncio.run(test())
