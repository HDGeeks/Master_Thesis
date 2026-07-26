import sys
import configparser
import os


def check_fixed_value(section_index, option_index, possible_values):
    """ Check if a configuration option has a valid value. A list of valid
    values is given as parameter. Error out if the value is invalid.

    :param section_index: The name of the configuration section.
    :param option_index: The name of the configuration option.
    :param possible_values: A list of valid values.
    """

    global config
    if config[section_index][option_index] not in possible_values:
        possible_values_string = ""
        for possible_value in possible_values:
            possible_values_string += possible_value + ", "
        possible_values_string = possible_values_string[:-2]
        print(f"Configuration error: The value of section \"{section_index}\" "
              f"and option \"{option_index}\" is not valid. Possible values "
              f"are: {possible_values_string}")
        sys.exit(1)


def check_positive_integer(section_index, option_index):
    """ Check if a configuration option is a positive integer (> 0). Error
    out if this is not the case.

    :param section_index: The name of the configuration section.
    :param option_index: The name of the configuration option.
    """
    global config
    error_message = f"Configuration error: The value of section \""\
                    + f"{section_index}\" and option \"{option_index}\" is "\
                    + f"not valid. Possible values are: An integer > 0"
    try:
        value = int(config[section_index][option_index])
        if value <= 0:
            print(error_message)
            sys.exit(1)
    except ValueError:
        print(error_message)
        sys.exit(1)


# Loading the default configuration file and overwriting it with the user
# configuration file.
config = configparser.ConfigParser()
tooldir = os.path.abspath(os.path.dirname(sys.argv[0]))
defaultfile = tooldir + "/config/config-defaults.ini"
configfile = tooldir + "/config/config.ini"
config.read(defaultfile)
if os.path.isfile(configfile):
    config.read(configfile)

# Checking the configuration options for the [general] section.
check_fixed_value("general", "debug", ["0", "1"])

# Checking the configuration options for the [example_dataset] section.
check_fixed_value("example_dataset", "creation_mode", ["all", "random", "ids"])
check_positive_integer("example_dataset", "num_random_publications")

# Checking the configuration options for the [dataset] section.
check_fixed_value("dataset", "has_evaluation_data", ["0", "1"])

# Checking the configuration options for the [gpt4all] section.
check_fixed_value("gpt4all", "device", ["cpu", "gpu"])
check_fixed_value("gpt4all",
                  "model_name",
                  ["Meta-Llama-3-8B-Instruct.Q4_0.gguf",
                   "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
                   "Phi-3-mini-4k-instruct.Q4_0.gguf",
                   "orca-mini-3b-gguf2-q4_0.gguf",
                   "gpt4all-13b-snoozy-q4_0.gguf"])

# Checking the configuration options for the [prompt] section.
check_positive_integer("prompt", "num_targets")
check_positive_integer("prompt", "targets_per_prompt")
check_positive_integer("prompt", "prompts_per_chat")
