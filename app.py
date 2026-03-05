from llama_cpp import Llama

llm = Llama(
    model_path="models/smollm-135m-q4.gguf",
    n_ctx=2048,
    n_threads=2
)

while True:
    prompt = input("You: ")

    output = llm(
        prompt,
        max_tokens=120,
        stop=["You:"]
    )

    print("AI:", output["choices"][0]["text"].strip())
