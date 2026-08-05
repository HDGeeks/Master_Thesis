
import json

with open("reference-repo/data/assets_example/metadata.json", "r") as f:
    publications = json.load(f)

for i in range(0,100):
    print(publications[i]["title"])
    print()