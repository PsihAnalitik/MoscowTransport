import os
import yaml

# ── Пути ─────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.yaml')

# ── Загрузка config.yaml ────────────────────────────────

def load_config(path: str = CONFIG_PATH) -> dict:
    """Загружает конфигурацию из YAML-файла."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Конфиг не найден: {path}\n"
            f"Скопируйте шаблон:  cp config.yaml.example config.yaml"
        )
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


CFG = load_config()

# ── Удобные аксессоры ───────────────────────────────────

DEFAULT_MODEL_KEY = CFG.get('default_model', next(iter(CFG.get('models', {}))))
MODELS = CFG.get('models', {})
EMBEDDING = CFG.get('embedding', {})
VOICE = CFG.get('voice', {})


def get_model_cfg(model_key: str | None = None) -> dict:
    """
    Возвращает словарь {'name', 'base_url', 'api_key', ...} для указанной модели.
    Если model_key не задан — берёт default_model из конфига.
    """
    key = model_key or DEFAULT_MODEL_KEY
    if key not in MODELS:
        available = ', '.join(MODELS.keys())
        raise KeyError(
            f"Модель '{key}' не найдена в config.yaml. "
            f"Доступные: {available}"
        )
    return MODELS[key]


if __name__ == '__main__':
    print("root directory:", ROOT_DIR)
    print("data directory:", DATA_DIR)
    print("config path:   ", CONFIG_PATH)
    print()
    print("default model: ", DEFAULT_MODEL_KEY)
    print("models:        ", list(MODELS.keys()))
    print()
    for key, m in MODELS.items():
        print(f"  [{key}]")
        print(f"    name:     {m['name']}")
        print(f"    base_url: {m['base_url']}")
        print(f"    context:  {m.get('context_window', '?')}")
    print()
    if EMBEDDING:
        print(f"embedding:  {EMBEDDING['name']}  →  {EMBEDDING['base_url']}")
    if VOICE:
        print(f"voice:      {VOICE['name']}")
        print(f"  http: {VOICE.get('http_url')}")
        print(f"  ws:   {VOICE.get('ws_url')}")