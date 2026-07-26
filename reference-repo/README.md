# Installation

We provide the nessesary commands to install this tool on Linux.

In order to use the code in this repository, you first need to have the following software installed on your system:

- Python 3.9 (or higher) with the virtual environment and pip modules

Then you need to clone it to your local machine and go into the cloned directory.

```
git clone https://github.com/paNeises/paper-2025-bigds.git
cd paper-2025-bigds
```

Next, you need to create a virtual environment, activate it and install the dependencies for the tool into it.

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

The tool is preconfigured to run the LLM's on the GPU. However, this configuration can be changed. The default values are stored in `config/config-default.ini`. You can overwrite these values by creating the file `config/config.ini` and put your preferred settings there. If you do want to use the CPU for whatever reason, you can create the file `config/config.ini` and add the following content to it:

```
[gpt4all]
device = cpu
```

This enables you to run the code on your local machine as long as the virtual environment is active.

# Running the evaluations discussed in the paper

To run the evaluations, that we discussed in the paper by yourself, you first need to re-create the dataset that we used. By running the following command with the default configuration, the tool downloads the whole papers part of the discussed D3 dataset, and extracts the 2500 publications that we used in our experiments:

```
python3 keywordextractor.py download-example-dataset
```

This creates the `data` directory and puts the example dataset in `data/assets_example`. The dataset consists of the following files:

- metadata.json: This file contains the list of all publications that are part of the dataset with their metadata.
- targets.json: This file contains the controlled vocabulary of out topics as a list.

The other files in the directory are just artifacts from the creation process that are not actively used. These can be deleted.

## Running the annotation and evaluation process based on titles

The tool comes preconfigured to run the annotation process based on titles on the example dataset that you created earlier. The results of the annotation process are stored in `data/assets_example/annotated_metadata.json`. This file then contains the full metadata from the `metadata.json` file together with a new entry called `keywordextractor_annotation` which contains the annotated topic in a list.

To run the annotation process based on titles, please run:

```
python3 keywordextractor.py annotate
```

After the results are written to `data/assets_example/annotated_metadata.json`, you can use the evaluation command in order to obtain the evaluation results:

```
python3 keywordextractor.py evaluate
```

This reads the previously computed file with the annotation results and does two seperate evaluations. Below we provide an example output and explain it.

```
Evaluating data for 2500 documents.
Evaluation of the number of found targets:
Avg: 0.7528
Min: 0
Max: 1
Evaluation of the number of correct targets according to the provided evaluation data:
Avg: 0.5024
Min: 0
Max: 1
```

The first evaluation correspond to the number of found targets. In order to compute these numbers, the tool loops over all publications and sums up the number of annotated topics (it should be noted that only topics can be annotated if they matched one of the entries in our controlled vocabulary). The tool then divides this number by the amount of publications and returns the obtained score as `Avg`. This number is the percentage of the cases success and misclassified discussed in the paper, since it gives us the percentage of documents, for which one topic from our controlled vocabulary was annotated. The remainder of publications must be of the case hallucination, since our matching process did not result in an annotated topic. The `Min` and `Max` values output the minimal/maximal number of annotated topics for a paper in the dataset and were introduced as additional information for the setting of annotating multiple topics to a publication. However the `Max` result of `1` shows that our algorithm does not annotate more topics to a publication then it was asked to (which is theoretically possible and would mess with the evaluation process).

The second evaluation correspons to the number of correct targets when compared with provided evaluation data. Here we also loop over all publications, but this time we only sum up the number of annotated topics of a pubication that are also in the evaluation data. We then also divide that number by the number of publications and provide it as `Avg` value. Thus we obtain the percentage of the success case this way. The `Min` and `Max` values again indicate the minimal/maximal number of correct topics for the publications and are not that important when we only want to annotate one topic.

## Running the annotation and evaluation process based on abstracts

The process of running the annotation is analogous to the process based on titles, except that you need to modify the configuration of the tool at two parameters first. In order to do that, you need to create the file `config/config.ini` with the following content (or append the content to it if it was created earlier in order to use the CPU):

```
[dataset]
document_index = abstract
[prompt]
document_name = abstract
```

Then we can run the same process as discussed for titles earlier. It should be noted that the tool warns you if the file `data/assets_example/annotated_metadata.json` exists and does nothing. So if you already performed the evaluation based on titles, you need to move the results of the annotation process in order to run a new annotation process.
