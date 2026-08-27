# OTUS. Локальный инференс и легкий деплой

### Блок 1. Установка и загрузка весов (15 мин)

Сначала подготовим оба движка. Ollama скачает модель сама, а для `llama.cpp` нам потребуется вручную скачать файл в формате GGUF.

**1. Подготовка Ollama**

```bash
# Установка Ollama (для Linux/Ubuntu)
curl -fsSL https://ollama.com/install.sh | sh

# Загрузка модели (3 миллиарда параметров, квантование по умолчанию Q4)
ollama pull llama3.2:3b
```

**2. Подготовка llama.cpp**
Чтобы не собирать C++ код с нуля на занятии, мы скачаем готовую модель с Hugging Face и будем использовать Python-биндинг `llama-cpp-python`, который включает в себя OpenAI-совместимый веб-сервер.

```bash
# Устанавливаем huggingface-cli как глобальный инструмент (для скачивания GGUF-файлов)
# (используем uv для скорости, если он установлен, или обычный pip)
uv tool install "huggingface_hub[cli]" --python 3.11.4

# Скачиваем GGUF-файл конкретной квантизации (Q4_K_M) в текущую папку models
mkdir models
hf download unsloth/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ./models

# устанавливаем llama-cpp-python с сервером в виртуальное окружение
uv add "llama-cpp-python[server]"
```

### Блок 2. Запуск серверов и Python-клиенты (20 мин)

Теперь поднимем два разных API-сервера и напишем скрипты для обращения к ним.

**1. Запуск серверов (в разных терминалах)**

*Терминал 1 (Ollama):*
Сервер Ollama обычно уже висит в фоне как `systemd` сервис (на порту 11434). Если нет, запускаем:

```bash
ollama serve
```

*Терминал 2 (llama.cpp):*
Запускаем сервер `llama.cpp`. Флаг `n_gpu_layers` (`-ngl`) переносит слои нейросети на видеокарту (если она есть).

```bash
python -m llama_cpp.server --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf --port 8080
```

**2. Python-клиент для Ollama (Свой API-формат)**

```python
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
```

**3. Python-клиент для llama.cpp (OpenAI-совместимый API)**

```python
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
```

### Блок 3. Настройка модели через Modelfile

Обычные запросы уже работают с базовой моделью `llama3.2:3b`. Теперь создадим
её вариант с фиксированными параметрами и системной инструкцией.

Файл `Modelfile` в корне проекта:

```text
FROM llama3.2:3b

PARAMETER temperature 0.3
PARAMETER num_ctx 4096

SYSTEM """
You are a helpful assistant for a local LLM inference workshop.
Answer clearly and concisely in the language of the user's question.
"""
```

Создадим модель и переключим Python-клиенты на неё:

```bash
ollama create otus-llama -f ./Modelfile
ollama list
```

В [src/ollama.py](src/ollama.py) и [src/router.py](src/router.py) замените значение глобальной
переменной `OLLAMA_MODEL` на `"otus-llama"`. После изменения `Modelfile` повторно
выполните `ollama create otus-llama -f ./Modelfile`.

### Блок 4. Профилирование ресурсов

**Команды для мониторинга:**

```bash
# Мониторинг VRAM (видеопамяти) в реальном времени, обновление каждую секунду
watch -n 1 nvidia-smi

# Мониторинг RAM и ядер CPU (для инференса без GPU)
htop
```

*На что смотреть:*

1. **Memory-Usage в `nvidia-smi`:** до и после отправки запроса (оценить размер загруженных весов + размер KV-кэша на контекст).
2. **Volatile GPU-Util:** прыгает ли утилизация до 100% при генерации. Если утилизация низкая, а генерация медленная - упор идет в пропускную способность памяти (Memory Bandwidth).

### Блок 5. Скрипт-роутер

Написать единый пайплайн, который маршрутизирует запрос. Если промпт короткий (быстрый вопрос) - шлем в Ollama. Если длинный контекст (анализ текста) - шлем в `llama.cpp` (так как там мы можем жестко контролировать `KV cache` параметры при запуске).

```python
import requests

OLLAMA_MODEL = "llama3.2:3b"

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

```