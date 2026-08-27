import requests
import time

OLLAMA_MODEL = "otus-llama"


def chat_with_ollama(prompt):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False # Установите True для потоковой выдачи
    }
    
    start_time = time.time()
    response = requests.post(url, json=payload).json()
    duration = time.time() - start_time
    
    # Ollama отдает удобную метрику eval_count / eval_duration
    tokens_per_sec = response['eval_count'] / (response['eval_duration'] / 1e9)
    
    print(f"[Ollama] Ответ: {response['message']['content'][:100]}...")
    print(f"[Ollama] Скорость: {tokens_per_sec:.2f} tokens/sec")


def main():
    chat_with_ollama("Объясни архитектуру Transformer в трех предложениях.")


if __name__ == "__main__":
    main()
