"""
load_model.py

Small, reusable loader for any local Hugging Face model directory
(e.g. the Qwen models under /localdata/dagstuhl/ai_models/huggingface-full/).

Usage examples:

    # one-shot generation
    python load_model.py \
        --model-path /localdata/dagstuhl/ai_models/huggingface-full/Qwen/Qwen3.5-9B \
        --prompt "Explain topic modeling in two sentences."

    # interactive chat loop
    python load_model.py \
        --model-path /localdata/dagstuhl/ai_models/huggingface-full/Qwen/Qwen3.5-9B \
        --chat

    # import and reuse in another script
    from load_model import load_model, generate
    model, tokenizer = load_model("/path/to/model")
    reply = generate(model, tokenizer, "Hello")
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path, dtype="auto", device_map="auto"):
    """
    Load a tokenizer and causal language model from a local directory.

    model_path : path to a folder containing config.json, tokenizer files,
                 and the .safetensors weight shards
    dtype      : "auto" lets transformers pick the dtype the weights were
                 saved in (usually bf16 for these Qwen checkpoints)
    device_map : "auto" lets transformers spread the model across whatever
                 GPUs are visible, falling back to CPU if none are found
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
    )

    # Put the model in evaluation mode since we are only doing inference,
    # not training (this disables dropout and similar training-only layers)
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=512):
    """
    Run one prompt through the model's chat template and return the reply
    as plain text.
    """
    messages = [{"role": "user", "content": prompt}]

    # apply_chat_template formats the prompt the way the model was trained
    # on (system/user/assistant tags), instead of feeding raw text
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # no_grad because we are not training, this saves memory during inference
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic output, change to True for variety
        )

    # slice off the input tokens so we only decode the newly generated part
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return reply


def chat_loop(model, tokenizer, max_new_tokens=512):
    """
    Simple multi-turn chat loop in the terminal. Keeps the full message
    history so the model has conversation context, type "exit" to quit.
    """
    history = []

    print("Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            break

        history.append({"role": "user", "content": user_input})

        input_ids = tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

        history.append({"role": "assistant", "content": reply})

        print(f"Model: {reply}\n")


def main():
    parser = argparse.ArgumentParser(description="Load and run any local Hugging Face model.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the local model directory, e.g. /localdata/.../Qwen/Qwen3.5-9B",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt to run once, ignored if --chat is set",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive multi-turn chat loop instead of a single prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per reply",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model, tokenizer = load_model(args.model_path)
    print("Model loaded.\n")

    if args.chat:
        chat_loop(model, tokenizer, max_new_tokens=args.max_new_tokens)
    elif args.prompt:
        reply = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)
        print(reply)
    else:
        print("Nothing to do: pass --prompt \"...\" for a single run, or --chat for an interactive loop.")


if __name__ == "__main__":
    main()