# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import base64
import os
from typing import Any, Optional

from openai import OpenAI


def get_image_base64_and_mime(image_path):
    """Convert image file to base64 string and get MIME type"""
    try:
        # Get MIME type based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")  # Default to JPEG

        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_data, mime_type
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None, None


def send_generate_request(
    messages,
    server_url=None,
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key=None,
    max_tokens=2048,  # Reduced default to be safer for models with smaller context windows
    tools=None,  # IGNORED - kept for backwards compatibility
):
    """
    Sends a request to the OpenAI-compatible API endpoint using the OpenAI client library.

    Args:
        server_url (str): The base URL of the server, e.g. "http://127.0.0.1:8000"
        messages (list): A list of message dicts, each containing role and content.
        model (str): The model to use for generation (default: "llama-4")
        api_key (str): API key for authentication (can be empty for some backends)
        max_tokens (int): Maximum number of tokens to generate (default: 2048, reduced for safety with smaller context windows)
        tools (list, optional): IGNORED - Tool definitions are embedded in system prompt instead.
                               This parameter is kept for backwards compatibility only.

    Returns:
        str: The generated response text from the server. Tool calls are parsed from text by agent_core.
    """
    # Process messages to convert image paths to base64
    processed_messages = []
    for message in messages:
        processed_message = message.copy()
        if message["role"] == "user" and "content" in message:
            processed_content = []
            for c in message["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    # Convert image path to base64 format
                    image_path = c["image"]

                    print("image_path", image_path)
                    if not image_path:
                        print("Warning: Empty image path in message content, skipping.")
                        continue
                    new_image_path = image_path.replace(
                        "?", "%3F"
                    )  # Escape ? in the path

                    # Read the image file and convert to base64
                    try:
                        base64_image, mime_type = get_image_base64_and_mime(
                            new_image_path
                        )
                        if base64_image is None:
                            print(
                                f"Warning: Could not convert image to base64: {new_image_path}"
                            )
                            continue

                        # Create the proper image_url structure with base64 data
                        processed_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high",
                                },
                            }
                        )

                    except FileNotFoundError:
                        print(f"Warning: Image file not found: {new_image_path}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing image {new_image_path}: {e}")
                        continue
                else:
                    processed_content.append(c)

            processed_message["content"] = processed_content
        processed_messages.append(processed_message)

    # Normalize server URL - ensure it ends with /v1 for OpenAI-compatible APIs
    if server_url and not server_url.endswith('/v1'):
        if server_url.endswith('/'):
            server_url = server_url + 'v1'
        else:
            server_url = server_url + '/v1'

    # Create OpenAI client with custom base URL
    client = OpenAI(api_key=api_key or "not-needed", base_url=server_url)

    try:
        print(f"🔍 Calling model {model}...")
        print(f"   Server URL: {server_url}")
        print(f"   Messages: {len(processed_messages)} messages")
        
        # Log message structure for debugging
        message_types = {}
        for msg in processed_messages:
            role = msg.get("role", "unknown")
            if "content" in msg:
                if isinstance(msg["content"], list):
                    content_types = [c.get("type", "unknown") for c in msg["content"] if isinstance(c, dict)]
                    message_types[role] = content_types
                else:
                    message_types[role] = type(msg["content"]).__name__
        print(f"   Message structure: {message_types}")
        
        # Prepare request parameters
        # NOTE: We intentionally DO NOT pass tools to the API request
        # because the OpenAI SDK automatically adds tool_choice="auto" when tools are present,
        # and vLLM without --enable-auto-tool-choice flag rejects this.
        # Instead, tool instructions are embedded in the system prompt and parsed from text response.
        request_params = {
            "model": model,
            "messages": processed_messages,
            "max_tokens": max_tokens,
            "n": 1,
        }
        
        # Add timeout to prevent hanging requests (2 minutes default)
        response = client.chat.completions.create(**request_params, timeout=120.0)

        # Extract the response
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            
            # Return text response - tool calls will be parsed from text by agent_core
            content = message.content
            
            # Handle case where content is None but tool_calls might be present
            if content is None:
                finish_reason = response.choices[0].finish_reason if response.choices else 'N/A'
                print(f"⚠️ Warning: Response content is None")
                print(f"   Model: {model}")
                print(f"   Server URL: {server_url}")
                print(f"   Finish reason: {finish_reason}")
                print(f"   Message count: {len(processed_messages)}")
                
                # Check for tool_calls - some models return tool_calls instead of content
                tool_calls_present = hasattr(message, 'tool_calls') and message.tool_calls
                if tool_calls_present:
                    print(f"   ⚠️ Tool calls present ({len(message.tool_calls)}): Model returned structured tool_calls instead of text")
                    print(f"   Tool calls: {message.tool_calls}")
                    print(f"   ⚠️ This model may be using structured tool calling, but agent expects text-based tool calls")
                    print(f"   💡 Possible solutions:")
                    print(f"      1. Check if the model supports text-based tool calling")
                    print(f"      2. Verify the model configuration and API compatibility")
                    print(f"      3. Check if tool_choice parameter needs to be set differently")
                    # Return None to trigger retry logic in agent_core
                    return None
                else:
                    print(f"   ❌ No content and no tool_calls - this indicates a model error or unexpected response format")
                    print(f"   💡 Possible causes:")
                    print(f"      1. Model server error or timeout")
                    print(f"      2. Model ran out of context or tokens")
                    print(f"      3. Invalid request format")
                    print(f"      4. Model configuration issue")
                    if finish_reason:
                        print(f"   Finish reason '{finish_reason}' may provide additional context")
                    return None
            return content
        else:
            print(f"❌ Unexpected response format: {response}")
            print(f"   Model: {model}")
            print(f"   Server URL: {server_url}")
            print(f"   Request params: model={request_params['model']}, max_tokens={request_params['max_tokens']}, n={request_params['n']}")
            print(f"   Message count: {len(processed_messages)}")
            print(f"   Response type: {type(response)}")
            print(f"   Response keys/attributes: {dir(response) if hasattr(response, '__dict__') else 'N/A'}")
            if hasattr(response, 'choices'):
                print(f"   Choices: {response.choices}")
                print(f"   Choices length: {len(response.choices) if response.choices else 0}")
            else:
                print(f"   Response has no 'choices' attribute")
            return None

    except Exception as e:
        import traceback
        # Special-case 404 to provide clearer guidance about base_url / endpoint
        try:
            from openai import NotFoundError
        except Exception:
            NotFoundError = None

        if NotFoundError and isinstance(e, NotFoundError):
            hint = (
                "Received 404 from LLM endpoint. Check that llm_config.base_url points to the LLM server "
                "(e.g., https://api.openai.com/v1 or your vLLM /v1 endpoint), not the SAM3 agent URL. "
                f"Current base_url: {server_url}"
            )
            print(f"❌ Request failed: {e} | {hint}")
            raise ValueError(hint) from e

        print(f"❌ Request failed: {e}")
        print(f"   Model: {model}")
        print(f"   Server URL: {server_url}")
        print(f"   Message count: {len(processed_messages)}")
        print(f"   Request params: model={request_params.get('model', 'N/A')}, max_tokens={request_params.get('max_tokens', 'N/A')}")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Full traceback: {traceback.format_exc()}")
        # Re-raise to see the actual error in logs
        raise


def send_direct_request(
    llm: Any,
    messages: list[dict[str, Any]],
    sampling_params: Any,
) -> Optional[str]:
    """
    Run inference on a vLLM model instance directly without using a server.

    Args:
        llm: Initialized vLLM LLM instance (passed from external initialization)
        messages: List of message dicts with role and content (OpenAI format)
        sampling_params: vLLM SamplingParams instance (initialized externally)

    Returns:
        str: Generated response text, or None if inference fails
    """
    try:
        # Process messages to handle images (convert to base64 if needed)
        processed_messages = []
        for message in messages:
            processed_message = message.copy()
            if message["role"] == "user" and "content" in message:
                processed_content = []
                for c in message["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        # Convert image path to base64 format
                        image_path = c["image"]
                        if not image_path:
                            print("Warning: Empty image path in message content, skipping.")
                            continue
                        new_image_path = image_path.replace("?", "%3F")

                        try:
                            base64_image, mime_type = get_image_base64_and_mime(
                                new_image_path
                            )
                            if base64_image is None:
                                print(
                                    f"Warning: Could not convert image: {new_image_path}"
                                )
                                continue

                            # vLLM expects image_url format
                            processed_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                }
                            )
                        except Exception as e:
                            print(
                                f"Warning: Error processing image {new_image_path}: {e}"
                            )
                            continue
                    else:
                        processed_content.append(c)

                processed_message["content"] = processed_content
            processed_messages.append(processed_message)

        print("🔍 Running direct inference with vLLM...")

        # Run inference using vLLM's chat interface
        outputs = llm.chat(
            messages=processed_messages,
            sampling_params=sampling_params,
        )

        # Extract the generated text from the first output
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            return generated_text
        else:
            print(f"Unexpected output format: {outputs}")
            return None

    except Exception as e:
        print(f"Direct inference failed: {e}")
        return None
