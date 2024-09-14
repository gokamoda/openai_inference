from dotenv import load_dotenv

from openai_inference import create_messsage, openai_inference

load_dotenv()

if __name__ == "__main__":
    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]

    messages = [
        create_messsage(
            system_message="You are a helpful assistant.",
            prompt=prompt,
        )
        for prompt in prompts
    ]

    results = openai_inference(
        messages, model_name="gpt-4o-mini", log_path="latest.log", max_tokens=1, batch=False
    )
    print(results)