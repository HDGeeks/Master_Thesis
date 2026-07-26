import sys


import modules.configuration as conf
import modules.data_processing as dp


def concatenate_targets_to_string_list(targets, count):
    """Concatenate all target strings, such that every list element is a
    concatenation of count target strings.

    :param targets: The targets list.
    :param count: How many targets should be concatenated per list item.
    """

    targets_split = [targets[i:i+count] for i in range(0, len(targets), count)]

    targets_string_list = []
    for split in targets_split:
        split_string = ""
        for target in split:
            split_string += target + ", "
        split_string = split_string[:-2]
        targets_string_list.append(split_string)

    return targets_string_list


def build_query_prompt_list(document, targets):
    """Build a sequence of prompts to annotate a given document.

    :param document: The document that should be annotated.
    :param targets: A list of possible targets used in the annotation.
    """

    document_name = conf.config["prompt"]["document_name"]
    target_name = conf.config["prompt"]["target_name"]
    num_targets = int(conf.config["prompt"]["num_targets"])
    targets_per_prompt = int(conf.config["prompt"]["targets_per_prompt"])

    targets_string_list = concatenate_targets_to_string_list(
        targets,
        targets_per_prompt)

    prompt_list = []

    prompt = "We want to create a list of " + target_name + "s in the "\
             + "following. We call this list targets_list."
    prompt_list.append(prompt)

    prompt = "Here are some " + target_name + "s that should be added to the "\
             + "targets_list: " + targets_string_list[0] + ". Please use the "\
             + "exact spelling that I provide to you."
    prompt_list.append(prompt)
    for i in range(1, len(targets_string_list)):
        prompt = "Here are some additional " + target_name + "s I want you "\
                 + "to add to the targets_list: " + targets_string_list[i]\
                 + ". Please use the exact spelling that I provide to you."
        prompt_list.append(prompt)

    prompt = "We now want to annotate a " + document_name + " with the "\
             + target_name + "s provided in the targets_list.\n"
    prompt += "Given the following " + document_name + ": " + document + "\n"
    if num_targets == 1:
        prompt += "Please assign 1 suitable " + target_name + " from the "\
                  + "targets_list to the " + document_name + ".\n"\
                  + "This " + target_name + " should be contained in the "\
                  + "targets_list we created earlier and use the exact "\
                  + "spelling of the " + target_name + " in the targets_list."\
                  + "\n"\
                  + "Please respond only with the 1 " + target_name\
                  + " without any further text."
    else:
        prompt += "Please assign up to " + str(num_targets) + " suitable "\
                  + target_name + "s from the targets_list to the "\
                  + document_name + ".\n"\
                  + "These " + target_name + "s should be contained in the "\
                  + "targets_list we created earlier and use the exact "\
                  + "spelling of the " + target_name + " in the targets_list."\
                  + "\n"\
                  + "Please respond only with the " + str(num_targets) + " "\
                  + target_name + "s separated by comma and without any "\
                  + "further text."
    prompt_list.append(prompt)

    return prompt_list


def build_query_prompt_lists(document, targets):
    targets_per_prompt = int(conf.config["prompt"]["targets_per_prompt"])
    prompts_per_chat = int(conf.config["prompt"]["prompts_per_chat"])
    targets_per_chat = targets_per_prompt * prompts_per_chat

    targets_per_chat_list = [targets[i:i+targets_per_chat] for i in
                             range(0, len(targets), targets_per_chat)]

    prompt_lists = []
    for targets_list in targets_per_chat_list:
        prompt_lists.append(build_query_prompt_list(document, targets_list))

    return prompt_lists


def process_query_prompt_lists(model, prompt_lists, targets_map):
    debug_mode = int(conf.config["general"]["debug"])
    annotated_targets = []
    for i in range(len(prompt_lists)):
        if debug_mode == 1:
            print(f"Running prompt list {i+1}/{len(prompt_lists)}.")
        prompt_list = prompt_lists[i]
        result = model.eval_prompt_list(prompt_list)
        if debug_mode == 1:
            print("Obtained the following result:")
            print(result)
        parsed_result = parse_result(result, targets_map)
        if debug_mode == 1:
            print("Extracted the following targets:")
            print(parsed_result)
        for target in parsed_result:
            annotated_targets.append(target)
    annotated_targets = list(set(annotated_targets))
    return annotated_targets


def parse_result(result, targets_set):
    """ Parse the answer to the annotation query generated by the LLM and
    extract the annotated targets.

    :param result: The query response generated by an LLM.
    :param targets_set: A set of possible target values in canonical form.
    """

    target_name = conf.config["prompt"]["target_name"]

    tokens = result.split(",")

    if len(tokens) == 1:
        print(f"Extracted {len(tokens)} {target_name} candidate from the LLM "
              f"response.")
    else:
        print(f"Extracted {len(tokens)} {target_name} candidates from the LLM "
              f"response.")

    for index in range(len(tokens)):
        tokens[index] = dp.build_canonical_target_form(tokens[index])

    final_tokens = []
    for token in tokens:
        if token in targets_set.keys():
            final_tokens.append(targets_set[token])

    if len(final_tokens) == 1:
        print(f"{len(final_tokens)} {target_name} candidate was found in the "
              f"given {target_name}s.")
    else:
        print(f"{len(final_tokens)} {target_name} candidates were found in "
              f"the given {target_name}s.")

    return final_tokens
