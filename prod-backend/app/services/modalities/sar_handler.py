def get_sar_payload(image_url: str, query_text: str) -> dict:
    """
    Determines the system prompt based on text intent and returns the full payload.
    This function is pure logic (no HTTP calls) and is used by the SARAdapter.
    """
    # 1. Determine Intent
    # Heuristic: if the user asks for a description/caption, use the captioning prompt.
    is_captioning = any(word in query_text.lower() for word in ["describe", "caption", "what is this image"])
    
    if is_captioning:
        system_prompt = "You are an expert in Synthetic Aperture Radar (SAR) imagery. Describe the radar backscatter, surface textures, and detected objects in this image."
    else:
        system_prompt = "You are an expert in Synthetic Aperture Radar (SAR) imagery. Answer the user's question by analyzing the radar signatures."

    # 2. Construct Payload
    return {
        "model": "sar-special",
        "messages": [
             {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query_text},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "temperature": 0.7 if is_captioning else 0.1
    }