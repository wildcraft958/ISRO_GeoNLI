import modal

# --- CONFIGURATION ---
# Point to the folder where we saved the merged model
MERGED_MODEL_PATH = "/data/models/merged-qwen-vl-captioning"
MODEL_NAME = "qwen-caption-special"  # The name clients will use

# Use the same volume as merge script
vol = modal.Volume.from_name("vlm-weights-merged-caption-special")

# Standard vLLM image with latest version
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.4",
        "transformers>=4.45.0",  # For AutoProcessor
        "huggingface_hub[hf_transfer]",
        "fastapi",
        "pillow",
        "httpx",
        "prometheus-fastapi-instrumentator"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("qwen3-vlm-captioning")

# Global LLM instance (cached per container)
_llm_cache = None

@app.function(
    image=image,
    gpu="A100-80GB",  # 40GB A100-80GB - sufficient for 8B models with FP16
    volumes={"/data/models": vol},
    timeout=600,  # 10 minutes per job
    scaledown_window=300,  # Keep container warm for 5 min
)
def process_inference_job(
    messages: list,
    image_bytes: bytes | None,
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    """Process a single inference job. LLM is cached per container."""
    import io

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    global _llm_cache
    
    # Load LLM once per container (cached in GPU memory)
    if _llm_cache is None:
        print("🔄 Loading LLM in job container...")
        _llm_cache = LLM(
            model=MERGED_MODEL_PATH,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.88,  # Reduced for 40GB A100-80GB safety margin
            tensor_parallel_size=1,
            enforce_eager=True,
            max_num_seqs=4,
        )
        print("✅ LLM loaded in job container")
    
    # Load processor once per container
    if not hasattr(process_inference_job, '_processor'):
        print("🔄 Loading processor in job container...")
        process_inference_job._processor = AutoProcessor.from_pretrained(
            MERGED_MODEL_PATH, 
            trust_remote_code=True
        )
        print("✅ Processor loaded in job container")
    
    llm = _llm_cache
    processor = process_inference_job._processor

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Convert image_bytes back to PIL Image if present
    image = None
    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes))

    # Format messages using processor's chat template
    try:
        formatted_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback: format manually with image placeholder
        formatted_prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                formatted_prompt += f"{role}: {content}\n"
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        formatted_prompt += f"{role}: {item.get('text', '')}\n"
                    elif item.get("type") == "image_url":
                        formatted_prompt += "<image>\n"  # Image placeholder for Qwen3-VL
        print(f"⚠️ Using fallback formatting: {e}")

    # vLLM expects prompt as dict with "prompt" and "multi_modal_data" keys for multimodal
    if image:
        outputs = llm.generate(
            [{"prompt": formatted_prompt, "multi_modal_data": {"image": image}}],
            sampling_params=sampling_params,
        )
    else:
        outputs = llm.generate(
            [formatted_prompt],
            sampling_params=sampling_params,
        )

    return outputs[0].outputs[0].text


@app.function(
    image=image,
    gpu="A100-80GB",  # 40GB A100-80GB - sufficient for 8B models with FP16
    volumes={"/data/models": vol},
    scaledown_window=300,
    # min_containers removed to stay within 2 A100-80GB limit (containers spin up on-demand)
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def serve():
    import base64
    import io

    import fastapi
    import httpx
    from fastapi.responses import JSONResponse
    from PIL import Image
    from prometheus_fastapi_instrumentator import Instrumentator
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    print("🚀 Initializing vLLM with Merged Vision Weights...")

    llm = LLM(
        model=MERGED_MODEL_PATH,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.88,  # Reduced for 40GB A100-80GB safety margin
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_seqs=4,
    )

    # Load processor for chat template formatting
    processor = AutoProcessor.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)

    print(f"✅ Engine ready! Serving as: {MODEL_NAME}")
    
    app = fastapi.FastAPI()

    # Instrument Prometheus metrics at /metrics
    Instrumentator().instrument(app).expose(app)

    http_client = httpx.AsyncClient(timeout=30)

    @app.on_event("shutdown")
    async def shutdown_http_client():
        await http_client.aclose()

    async def extract_messages_and_image(messages):
        """Extract messages in OpenAI format and image as PIL Image."""
        processed_messages = []
        image = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle string content (text-only)
            if isinstance(content, str):
                processed_messages.append({
                    "role": role,
                    "content": content
                })
                continue

            # Handle list content (multimodal)
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "text":
                        new_content.append(item)
                    elif item.get("type") == "image_url" and image is None:
                        # Extract image
                        image_url = item.get("image_url", {}).get("url", "")
                        if not image_url:
                            continue
                        
                        if image_url.startswith("data:"):
                            # Base64 image
                            _, b64_data = image_url.split(",", 1)
                            image_bytes = base64.b64decode(b64_data)
                            image = Image.open(io.BytesIO(image_bytes))
                        else:
                            # URL image
                            resp = await http_client.get(image_url)
                            resp.raise_for_status()
                            image = Image.open(io.BytesIO(resp.content))
                        
                        # Keep image_url in content for vLLM to process
                        new_content.append(item)
                
                processed_messages.append({
                    "role": role,
                    "content": new_content
                })
        
        return processed_messages, image

    @app.post("/v1/chat/completions")
    async def chat(request: fastapi.Request):
        try:
            req = await request.json()
            messages = req.get("messages", [])
            model = req.get("model", MODEL_NAME)
            max_tokens = req.get("max_tokens", 512)
            temperature = req.get("temperature", 0.7)

            # Extract messages in OpenAI format and image
            processed_messages, image = await extract_messages_and_image(messages)

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Format messages using processor's chat template (handles image placeholders)
            try:
                formatted_prompt = processor.apply_chat_template(
                    processed_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback: format manually with image placeholder
                formatted_prompt = ""
                for msg in processed_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        formatted_prompt += f"{role}: {content}\n"
                    elif isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                formatted_prompt += f"{role}: {item.get('text', '')}\n"
                            elif item.get("type") == "image_url":
                                formatted_prompt += "<image>\n"  # Image placeholder for Qwen3-VL
                print(f"⚠️ Using fallback formatting: {e}")

            # vLLM expects prompt as dict with "prompt" and "multi_modal_data" keys for multimodal
            if image:
                outputs = llm.generate(
                    [{"prompt": formatted_prompt, "multi_modal_data": {"image": image}}],
                    sampling_params=sampling_params,
                )
            else:
                outputs = llm.generate(
                    [formatted_prompt],
                    sampling_params=sampling_params,
                )

            generated_text = outputs[0].outputs[0].text

            # Estimate prompt tokens from messages (rough approximation)
            prompt_text = ""
            for msg in processed_messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    prompt_text += content + " "
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            prompt_text += item.get("text", "") + " "
            
            prompt_tokens = len(prompt_text.split()) if prompt_text else 0
            completion_tokens = len(generated_text.split())

            return JSONResponse(
                {
                    "id": "chatcmpl-modal",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": generated_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )

        except Exception as e:
            import traceback

            return JSONResponse(
                {"error": str(e), "traceback": traceback.format_exc()},
                status_code=500,
            )

    @app.post("/v1/chat/completions/async")
    async def chat_async(request: fastapi.Request):
        """Submit an async inference job and return call_id for polling."""
        try:
            req = await request.json()
            messages = req.get("messages", [])
            model = req.get("model", MODEL_NAME)
            max_tokens = req.get("max_tokens", 512)
            temperature = req.get("temperature", 0.7)

            # Extract messages and image
            processed_messages, image = await extract_messages_and_image(messages)
            
            # Convert PIL Image to bytes for serialization
            image_bytes = None
            if image:
                import io
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()

            # Look up the deployed function
            try:
                inference_job = modal.Function.from_name(
                    "qwen3-vlm-captioning", "process_inference_job"
                )
            except Exception as e:
                return JSONResponse(
                    {
                        "error": f"Function not found. Make sure to deploy with 'modal deploy deploy_vlm.py'. Error: {str(e)}"
                    },
                    status_code=503,
                )

            # Spawn the job - pass messages and image_bytes
            call = await inference_job.spawn.aio(
                messages=processed_messages,
                image_bytes=image_bytes,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return JSONResponse(
                {
                    "call_id": call.object_id,
                    "status": "pending",
                    "model": model,
                }
            )

        except Exception as e:
            import traceback

            return JSONResponse(
                {"error": str(e), "traceback": traceback.format_exc()},
                status_code=500,
            )

    @app.get("/v1/chat/completions/status/{call_id}")
    async def get_job_status(call_id: str):
        """Poll for job result. Returns 202 if pending, 200 if complete, 404 if expired."""
        try:
            function_call = modal.FunctionCall.from_id(call_id)

            try:
                # Poll non-blocking (timeout=0)
                result = await function_call.get.aio(timeout=0)
                generated_text = result

                # Return OpenAI-compatible response
                return JSONResponse(
                    {
                        "id": f"chatcmpl-{call_id}",
                        "object": "chat.completion",
                        "model": MODEL_NAME,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": generated_text,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,  # Would need to track this separately
                            "completion_tokens": len(generated_text.split()),
                            "total_tokens": len(generated_text.split()),
                        },
                    }
                )

            except modal.exception.OutputExpiredError:
                return JSONResponse(
                    {"error": "Job result expired (older than 7 days)"},
                    status_code=404,
                )
            except TimeoutError:
                # Job still pending
                return JSONResponse(
                    {"status": "pending", "call_id": call_id}, status_code=202
                )

        except Exception as e:
            import traceback

            return JSONResponse(
                {"error": str(e), "traceback": traceback.format_exc()},
                status_code=500,
            )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_NAME}

    return app
