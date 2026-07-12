import asyncio
import httpx
import base64
import os

async def test_modal_real():
    # Read the real manga test image
    img_path = r"c:\Users\chevi\demo-mimo\manga-translator-api\manga_test.jpg"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    
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
            print(f"Sending real image ({len(b64)} bytes) to {url}...")
            resp = await client.post(url, json=payload, timeout=120.0)
            print("Status:", resp.status_code)
            if resp.status_code == 200:
                data = resp.json()
                print(f"Success! Translated regions: {data.get('region_count')}")
                print(f"Processing time: {data.get('processing_time_ms')} ms")
            else:
                print("Response:", resp.text)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_modal_real())
