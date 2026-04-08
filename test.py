from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    print("loading qwen3-32b...")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    model.tokenizer = tokenizer
    prompt = "What is the capital of France?"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print(model.tokenizer.decode(outputs[0], skip_special_tokens=True))