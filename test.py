from PIL import Image
import requests
from io import BytesIO
from torch.nn.functional import cosine_similarity
import torch
from transformers import AutoProcessor
from vdocrag.vdocretriever.modeling import VDocRetriever
from vdocrag.vdocgenerator.modeling import VDocGenerator

### Retrieval ###

processor = AutoProcessor.from_pretrained('microsoft/Phi-3-vision-128k-instruct', trust_remote_code=True)
model = VDocRetriever.load('microsoft/Phi-3-vision-128k-instruct', lora_name_or_path='NTT-hil-insight/VDocRetriever-Phi3-vision', pooling='eos', normalize=True, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, use_cache=False).to('cuda:0')

# Process query inputs and get embeddings
queries = ["Instruct: I’m looking for an image that answers the question.\nQuery: What is the total percentage of Palestinians residing at West Bank?</s>", 
           "Instruct: I’m looking for an image that answers the question.\nQuery: How many international visitors came to Japan in 2017?</s>"]
query_inputs = processor(queries, return_tensors="pt", padding="longest", max_length=256, truncation=True).to('cuda:0')

with torch.no_grad():
    model_output = model(query=query_inputs, use_cache=False)
    query_embeddings = model_output.q_reps

# List of image URLs
urls = [
    "https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA/resolve/main/image1.png",
    "https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA/resolve/main/image2.png"
]

# Download, open, and resize images
doc_images = [Image.open(BytesIO(requests.get(url).content)).resize((1344, 1344)) for url in urls]

# Process images with prompt
doc_prompt = "<|image_1|>\nWhat is shown in this image?</s>"
collated_list = [
    processor(
        doc_prompt,
        images=image,
        return_tensors="pt",
        padding="longest",
        max_length=4096,
        truncation=True
    ).to('cuda:0') for image in doc_images
]

# Stack tensors into input dict
doc_inputs = {
    key: torch.stack([item[key][0] for item in collated_list], dim=0)
    for key in ['input_ids', 'attention_mask', 'pixel_values', 'image_sizes']
}

with torch.no_grad():
    model_output = model(document=doc_inputs, use_cache=False)
    doc_embeddings = model_output.p_reps

# Calculate cosine similarity
num_queries = query_embeddings.size(0)
num_passages = doc_embeddings.size(0)

for i in range(num_queries):
    query_embedding = query_embeddings[i].unsqueeze(0)
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    print(f"Similarities for Query {i}: {similarities.cpu().float().numpy()}")

# >> Similarities for Query 0: [0.5078125  0.38085938]
#    Similarities for Query 1: [0.37695312 0.5703125 ]

### Generation ###

model = VDocGenerator.load('microsoft/Phi-3-vision-128k-instruct', lora_name_or_path='NTT-hil-insight/VDocGenerator-Phi3-vision', trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, use_cache=False).to('cuda:0')

query =  "How many international visitors came to Japan in 2017? \n Answer briefly."

image_tokens = "\n".join([f"<|image_{i+1}|>" for i in range(len(doc_images))])
messages = [{"role": "user", "content": f"{image_tokens}\n{query}"}]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

processed = processor(prompt, images=doc_images, return_tensors="pt").to('cuda:0')
generate_ids = model.generate(processed, generation_args={"max_new_tokens": 64, "temperature": 0.0, "do_sample": False, "eos_token_id": processor.tokenizer.eos_token_id})
generate_ids = generate_ids[:, processed['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
response = response.strip()
print("Model prediction: {0}".format(response))

## >> Model prediction: 28.69m