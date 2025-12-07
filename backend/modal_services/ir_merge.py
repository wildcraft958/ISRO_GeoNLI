import modal

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER_REPO_ID = "pookie0345/Monster-IR-1"
MERGED_MODEL_DIR = "/data/models/merged-qwen3vl-ir"

# Create a fresh volume for merged model
vol = modal.Volume.from_name("vlm-weights-merge-qwen3vl-ir", create_if_missing=True)

# We need recent transformers/peft for Qwen3-VL
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.45.0",
        "peft",
        "torch",
        "torchvision",
        "huggingface_hub",
        "accelerate",
        "qwen-vl-utils"  # Required for Qwen3-VL
    )
)

app = modal.App("merge-qwen3-vl-ir")

@app.function(
    image=image,
    volumes={"/data/models": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def merge_model():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from huggingface_hub import login, upload_folder

    
    # 1. Load Base Model
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"🔄 Loading Adapter: {ADAPTER_REPO_ID}...")
    
    # 2. Load Adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO_ID)
    
    print("🔄 Merging weights (this fixes the vLLM vision limitation)...")
    
    # 3. Merge Vision + Text weights permanently
    merged_model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to {MERGED_MODEL_DIR}...")
    
    # 4. Save the Merged Model
    merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
    
    # 5. Save the Processor/Tokenizer (CRITICAL for added tokens)
    print("💾 Saving processor and tokenizer...")
    processor = AutoProcessor.from_pretrained(
        ADAPTER_REPO_ID, 
        trust_remote_code=True,
        fix_mistral_regex=True  # Apply regex fix
    )
    processor.save_pretrained(MERGED_MODEL_DIR)
    
    # Commit the volume to persist changes
    vol.commit()
    
    print("✅ Merge Complete! You can now serve this path.")