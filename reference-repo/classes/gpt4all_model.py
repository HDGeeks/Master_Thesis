import sys


import gpt4all


import modules.configuration as conf


class gpt4all_model:

    def __init__(self):
        model_name = conf.config["gpt4all"]["model_name"]
        device = conf.config["gpt4all"]["device"]
        self.model = gpt4all.GPT4All(model_name, device=device)

    def eval_prompt_list(self, prompt_list):
        debug_mode = int(conf.config["general"]["debug"])
        with self.model.chat_session():
            current_index = 1
            for index in range(len(prompt_list)):
                if debug_mode == 1:
                    print(f"Running prompt {current_index}/"
                          f"{len(prompt_list)}.")
                result = self.model.generate(prompt_list[index])
                current_index += 1
        return result
