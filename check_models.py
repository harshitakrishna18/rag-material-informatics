from huggingface_hub import InferenceClient
import os

client = InferenceClient(token=os.environ.get("HF_TOKEN"))

# Returns a dict keyed by provider (e.g. "hf-inference") with model lists
models = client.list_deployed_models()

hf_models = models.get("hf-inference", [])
print("hf-inference deployed models:", len(hf_models))

# Filter likely chat/generative models (rough heuristic)
for m in hf_models:
    if any(x in m.lower() for x in ["instruct", "chat", "llama", "mistral", "smollm", "phi", "qwen"]):
        print(m)
