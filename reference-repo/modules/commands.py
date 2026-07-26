import sys
import os
import random
import json
from datetime import datetime


import requests
from tqdm import tqdm

import modules.configuration as conf
import modules.data_fetching as df
import modules.data_loading as dl
import modules.data_processing as dp
import modules.prompt_engineering as pe
import modules.evaluation as ev
import classes.gpt4all_model as gpt4all_model


def download_example_dataset():
    """Download an example dataset for the tool.

    This function downloads and extracts an example dataset for the tool. It
    uses the D3 dataset (https://zenodo.org/records/7071698) as base and
    extracts all publication records that have title, abstract and cso concept
    classification information (the dataset fields semantic, syntactic, union
    and enhanced). It then aggregates all used cso concepts to provide a
    targets list. For more information about the cso concepts see
    https://github.com/jpwahle/lrec22-d3-dataset/issues/1 and the there
    mentioned resources.
    """

    creation_mode = conf.config["example_dataset"]["creation_mode"]
    num_random_publications =\
        int(conf.config["example_dataset"]["num_random_publications"])
    id_list_json = conf.config["example_dataset"]["id_list_json"]

    data_dir = os.path.join(conf.tooldir, "data")
    if os.path.exists(data_dir) and not os.path.isdir(data_dir):
        print(f"Error: The path {data_dir} exists and is not a directory. "
              f"This path is used in order to store the necessary datasets "
              f"for the tool. Please move the contents of this path to "
              f"another place and run the tool again.")
        sys.exit(1)
    elif not os.path.isdir(data_dir):
        os.mkdir(data_dir, 0o775)

    example_data_dir = os.path.join(data_dir, "assets_example")
    if os.path.exists(example_data_dir):
        print(f"Error: The directory {example_data_dir} sees to exist. This "
              f"directory is used for the example dataset. If you already "
              f"have data in this path, please move it to another place "
              f"before downloading the example dataset.")
        sys.exit(1)
    os.mkdir(example_data_dir, 0o775)

    print("Downloading the papers part of the D3 dataset.")
    d3_papers_dataset_url = "https://zenodo.org/records/7071698/files/"\
                            + "2022-11-30-papers.jsonl.gz?download=1"
    d3_papers_path = os.path.join(example_data_dir, "d3_papers.jsonl")
    response = requests.get(d3_papers_dataset_url, stream=True)
    total_filesize = int(response.headers.get('content-length', 0))
    block_size_in_kb = 1024
    progress_bar = tqdm(total=total_filesize, unit='iB', unit_scale=True)
    d3_papers_gz_path = os.path.join(example_data_dir, "d3_papers.jsonl.gz")
    with open(d3_papers_gz_path, "wb") as gz_file:
        for data in response.iter_content(block_size_in_kb):
            progress_bar.update(len(data))
            gz_file.write(data)
    progress_bar.close()

    print("Extracting the downloaded papers part of the D3 dataset from the "
          ".gz file.")
    thread_count = os.cpu_count() - 1
    with open(d3_papers_path, "wb") as jsonlfile:
        jsonlfile.write(df.extract_gzip(d3_papers_gz_path,
                                        threads=thread_count))

    subjects = ["artificial intelligence",
                "computer aided design",
                "computer hardware",
                "computer imaging and vision",
                "computer networks",
                "computer programming",
                "computer security",
                "computer systems",
                "data mining",
                "human computer interaction",
                "information retrieval",
                "information technology",
                "internet",
                "operating systems",
                "robotics",
                "software",
                "software engineering",
                "theoretical computer science",
                "bioinformatics"]

    print("Extracting the relevant publication metadata from the D3 dataset.")
    publications = []
    with open(d3_papers_path, "r") as jsonlfile:
        line_sum = sum(1 for line in jsonlfile)
    with open(d3_papers_path, "r") as jsonlfile:
        for line in tqdm(jsonlfile, total=line_sum):
            line_data = json.loads(line)

            line_corpusid = None
            line_title = None
            line_abstract = None
            line_cso_concepts_set = set()

            if "corpusid" in line_data.keys():
                line_corpusid = line_data["corpusid"]
            if "title" in line_data.keys():
                line_title = line_data["title"]
            if "abstract" in line_data.keys():
                line_abstract = line_data["abstract"]
            if "syntactic" in line_data.keys():
                for cso_concept in line_data["syntactic"]:
                    line_cso_concepts_set.add(cso_concept)
            if "semantic" in line_data.keys():
                for cso_concept in line_data["semantic"]:
                    line_cso_concepts_set.add(cso_concept)
            if "union" in line_data.keys():
                for cso_concept in line_data["union"]:
                    line_cso_concepts_set.add(cso_concept)
            if "enhanced" in line_data.keys():
                for cso_concept in line_data["enhanced"]:
                    line_cso_concepts_set.add(cso_concept)
            line_subjects = []
            for subject in subjects:
                if subject in line_cso_concepts_set:
                    line_subjects.append(subject)

            if len(line_subjects) > 0 and line_title is not None\
                    and line_abstract is not None\
                    and line_corpusid is not None:
                publication_metadata = {"D3 ID": line_corpusid,
                                        "title": line_title,
                                        "abstract": line_abstract,
                                        "subjects": line_subjects}
                publications.append(publication_metadata)

    if creation_mode == "random":
        if num_random_publications < len(publications):
            print(f"Extracting {num_random_publications} publications from "
                  f"the D3 dataset at random.")
            random.shuffle(publications)
            publications = publications[:num_random_publications]
        elif num_random_publications == len(publications):
            print("The number of wanted publications is equal to the number "
                  "of valid publications in the D3 dataset, using all "
                  "available publications instead.")
        else:
            print("The number of wanted publications exceeds the number of "
                  "valid publications in the D3 dataset, using all available "
                  "publications instead.")
    elif creation_mode == "ids":
        print(f"Extracting publications based on the given corpus id list.")
        with open(id_list_json, "r") as jsonfile:
            filter_id_set = set(json.load(jsonfile))
        filtered_publications = []
        for publication in publications:
            if publication["D3 ID"] in filter_id_set:
                filtered_publications.append(publication)
        publications = filtered_publications
        del filtered_publications

    print("Sorting the publications based on their D3 ID.")
    publications = sorted(publications, key=lambda x: x["D3 ID"])

    print("Creating the publications D3 id list.")
    publications_id_list = []
    for publication in publications:
        publications_id_list.append(publication["D3 ID"])

    print("Saving the publication metadata, the subjects and the ids lists.")
    example_data_metadata_path = os.path.join(example_data_dir,
                                              "metadata.json")
    with open(example_data_metadata_path, "w", encoding="utf-8") as jsonfile:
        json.dump(publications, jsonfile, indent=2)
    example_data_targets_path = os.path.join(example_data_dir,
                                             "targets.json")
    with open(example_data_targets_path, "w", encoding="utf-8") as jsonfile:
        json.dump(subjects, jsonfile, indent=2)
    example_data_id_list_path = os.path.join(example_data_dir,
                                             "used_D3_ID_list.json")
    with open(example_data_id_list_path, "w", encoding="utf-8") as jsonfile:
        json.dump(publications_id_list, jsonfile, indent=2)


def annotate():
    """Annotates the configured dataset with the given target values.

    This function reads in the given dataset and tries to annotate it with
    the given target values by using an LLM. The list of annotated targets is
    only allowed to contain the provided targets. Any output from the LLM that
    does not match a provided target string is discarded. The annotated dataset
    is the saved in the configured path.
    """

    debug_mode = int(conf.config["general"]["debug"])
    annotated_file = dl.get_abspath(conf.config["dataset"]["annotated_json"])
    publication_document_index = conf.config["dataset"]["document_index"]
    publication_annotation_index = conf.config["dataset"]["annotation_index"]
    has_evaluation_data = int(conf.config["dataset"]["has_evaluation_data"])
    publication_evaluation_data_index =\
        conf.config["dataset"]["evaluation_data_index"]
    num_targets = int(conf.config["prompt"]["num_targets"])

    if os.path.exists(annotated_file):
        print(f"The file {annotated_file} exists, which would be user for the "
              f"annotated dataset. Please move the current file in order to "
              f"create a new annotation. Exiting!")
        sys.exit(1)

    model = gpt4all_model.gpt4all_model()

    publications, targets = dl.load_data()

    targets_map = dp.build_canonical_targets_map(targets)

    start_time = datetime.now()

    num_publications = len(publications)
    current_publication = 1
    for publication in publications:
        print(f"Processing publication {current_publication}/"
              f"{num_publications}:")

        document = publication[publication_document_index]
        prompt_lists = pe.build_query_prompt_lists(document, targets)

        if debug_mode == 1:
            print("Using the following prompt lists:")
            print(prompt_lists)

        annotated_targets = pe.process_query_prompt_lists(model,
                                                          prompt_lists,
                                                          targets_map)

        while len(annotated_targets) > num_targets:
            if debug_mode == 1:
                print("The document needs another iteration of prompts.")

            prompt_lists = pe.build_query_prompt_lists(document,
                                                       annotated_targets)

            if debug_mode == 1:
                print("Using the following prompt lists:")
                print(prompt_lists)

            annotated_targets = pe.process_query_prompt_lists(model,
                                                              prompt_lists,
                                                              targets_map)

        if debug_mode == 1:
            print("Extracted the following targets:")
            print(annotated_targets)

        publication[publication_annotation_index] = annotated_targets

        if has_evaluation_data == 1:
            matching_targets = 0
            for target in publication[publication_annotation_index]:
                if target in publication[publication_evaluation_data_index]:
                    matching_targets += 1
            print(f"{matching_targets} of these could be found in the "
                  f"evaluation data.")

        current_publication += 1

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"The annotation of all documents took {duration} time")

    with open(annotated_file, "w", encoding="utf-8") as jsonfile:
        json.dump(publications, jsonfile, indent=2)


def evaluate():
    """Evaluation function for an annotated dataset.

    This function reads the configured annotated dataset created by the
    annotate() function and generates evaluation data by matching the
    generated targets with the given evaluation targets in the dataset.
    """

    debug_mode = int(conf.config["general"]["debug"])
    has_evaluation_data = int(conf.config["dataset"]["has_evaluation_data"])
    publication_annotation_index = conf.config["dataset"]["annotation_index"]
    publication_evaluation_data_index =\
        conf.config["dataset"]["evaluation_data_index"]

    if has_evaluation_data != 1:
        print(f"The configuration implies that no evaluation data exists in "
              f"the dataset. Evaluation is not possible. Exiting!")
        sys.exit(1)

    annotated_metadata = dl.load_annotated_metadata()
    publication_count = len(annotated_metadata)
    if publication_count == 0:
        print(f"There is no metadata in the provided annotated json file. "
              f"Exiting!")
        sys.exit(1)
    print(f"Evaluating data for {publication_count} documents.")

    avg_value, min_value,  max_value = ev.build_targets_count_stats(
        annotated_metadata,
        publication_annotation_index)
    print(f"Evaluation of the number of found targets:")
    print(f"Avg: {avg_value}")
    print(f"Min: {min_value}")
    print(f"Max: {max_value}")

    avg_value, min_value,  max_value = ev.build_matching_targets_count_stats(
        annotated_metadata,
        publication_annotation_index,
        publication_evaluation_data_index)
    print(f"Evaluation of the number of correct targets according to the "
          f"provided evaluation data:")
    print(f"Avg: {avg_value}")
    print(f"Min: {min_value}")
    print(f"Max: {max_value}")
