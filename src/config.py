# src/config.py
import os

from dotenv import load_dotenv


def configure_api():
    """
    Loads API keys from environment variables or .env file.
    Supports Together AI and OpenRouter providers.
    """
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path,override=True)

    # Together AI — primary and fallback judge models
    together_key = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHER_API")
    if together_key:
        os.environ["TOGETHER_API_KEY"] = together_key
        print("Together AI API key configured successfully.")
    else:
        print("Warning: TOGETHER_API_KEY not found. Set it in .env file.")

    # Anthropic — primary judge model (Opus 4.8) for the auditor + rubric scorer
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        print("Anthropic API key configured (Opus judge enabled).")
    else:
        print("Warning: ANTHROPIC_API_KEY not found. The Opus judge will fall back "
              "to Together models. Set it in .env to use Opus 4.8.")

    # OpenRouter — optional, used if OpenRouter models are specified
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
        print("OpenRouter API key configured successfully.")

    # OpenAI direct — optional
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        print("OpenAI API key configured (direct access enabled).")

    if not together_key and not openrouter_key:
        print("Warning: No API keys found. Set TOGETHER_API_KEY in .env file.")