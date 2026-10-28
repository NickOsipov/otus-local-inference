import requests

OLLAMA_MODEL = "otus-llama-practice:latest"


def smart_router(prompt):
    # Если промпт длиннее 200 символов, используем llama.cpp (на порту 8080)
    # Иначе используем Ollama (на порту 11434)
    
    if len(prompt) > 200:
        print("-> Роутинг в llama.cpp (тяжелый запрос)")
        url = "http://localhost:8080/v1/chat/completions"
        payload = {"messages": [{"role": "user", "content": prompt}]}
        response = requests.post(url, json=payload).json()
        return response['choices'][0]['message']['content']
    else:
        print("-> Роутинг в Ollama (легкий запрос)")
        url = "http://localhost:11434/api/chat"
        payload = {"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}
        response = requests.post(url, json=payload).json()
        return response['message']['content']

# Тестируем
short_prompt = "Что такое RAG?"
long_prompt = "Проанализируй следующий текст и сделай выжимку: " + ("текст " * 50)

print(smart_router(short_prompt))
print("-" * 50)
print(smart_router(long_prompt))
