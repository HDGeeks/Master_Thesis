import sys


import modules.commands as commands


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No command specified. Exiting!")
        sys.exit(1)
    command = sys.argv[1]
    if command == "download-example-dataset":
        commands.download_example_dataset()
    elif command == "annotate":
        commands.annotate()
    elif command == "evaluate":
        commands.evaluate()
    else:
        print("Error: Specified command was not found. Exiting!")
        sys.exit(1)
