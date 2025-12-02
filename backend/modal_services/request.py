import asyncio
import base64
from pathlib import Path
from openai import AsyncOpenAI

# Replace with your Modal URL
API_URL = "https://maximuspookus--qwen3-vlm-vqa-serve.modal.run/v1"
IMAGE_PATH = Path(__file__).parent / "sample1.png"


def load_image_b64() -> str:
    with open(IMAGE_PATH, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def main():
    import time
    
    # Create client with longer timeout for large satellite images (5 minutes)
    client = AsyncOpenAI(
        base_url=API_URL, 
        api_key="EMPTY",
        timeout=300.0  # 5 minutes timeout
    )
    base64_image = load_image_b64()

    print("Sending Image to VQA Model...")
    print(f"Image size: {len(base64_image)} base64 chars")
    start_time = time.time()

    try:
        response = await client.chat.completions.create(
            model="qwen-vqa-special",  # Must match MODEL_NAME in deploy code
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this satellite image? Describe the key features and answer any questions about the content."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.7,
        )

        elapsed = time.time() - start_time
        print(f"✅ Response received in {elapsed:.2f} seconds")
        print("Response:", response.choices[0].message.content)
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f} seconds: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())