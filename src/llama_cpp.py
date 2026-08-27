import requests
import time

def chat_with_llama_cpp(prompt):
    url = "http://localhost:8080/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    start_time = time.time()
    response = requests.post(url, json=payload).json()
    duration = time.time() - start_time
    
    # Расчет скорости вручную, так как OpenAI API не отдает точное время генерации
    generated_tokens = response['usage']['completion_tokens']
    tokens_per_sec = generated_tokens / duration
    
    print(f"[llama.cpp] Ответ: {response['choices'][0]['message']['content'][:200]}...")
    print(f"[llama.cpp] Примерная скорость: {tokens_per_sec:.2f} tokens/sec")


def main():
    chat_with_llama_cpp("Объясни архитектуру Transformer в трех предложениях.")

if __name__ == "__main__":
    main()
