def build_targets_count_stats(annotated_metadata,
                              publication_annotation_index):
    """Compute statistics about the number of targets in the annotated data
    that were actually in the target list provided.

    Goes through the list of all annotated publications and returns the minimum,
    the maximum and the average number of targets in the annotated data. This
    data comes from the annotation process with the LLM and can only contain
    targets that were provided in the targets list. All non-matching targets
    are rejected in the annotation process.

    :param annotated_metadata: The annotated metadata, that should be
    evaluated.
    :param publication_annotation_index: The index name of the annotated data
    in the annotated_metadata.
    :return avg_value: The average number of targets per publication in the
    annotated data.
    :return: min_value: The minimum number of targets per publication in the
    annotated data.
    :return: max_value: The maximum number of targets per publication in the
    annotated data.
    """

    publication_count = len(annotated_metadata)
    min_value = len(annotated_metadata[0][publication_annotation_index])
    max_value = len(annotated_metadata[0][publication_annotation_index])
    value_sum = 0

    for publication in annotated_metadata:
        value = len(publication[publication_annotation_index])
        if value > max_value:
            max_value = value
        if value < min_value:
            min_value = value
        value_sum += value

    avg_value = value_sum / publication_count

    return avg_value, min_value, max_value


def build_matching_targets_count_stats(annotated_metadata,
                                       publication_annotation_index,
                                       publication_evaluation_data_index):
    """Compute statistics about the number of targets returned by the LLM
    that are in the evaluation data given in the dataset.

    Goes through the list of all annotated publications and returns the minimum,
    the maximum and the average number of targets in the annotated data that
    can be found in the evaluation data.

    :param annotated_metadata: The annotated metadata, that should be
    evaluated.
    :param publication_annotation_index: The index name of the annotated data
    in the annotated_metadata.
    :param publication_evaluation_data_index: The index name of the evaluation
    data in the annotated_metadata.
    :return avg_value: The average number of targets per publication in the
    annotated data.
    :return: min_value: The minimum number of targets per publication in the
    annotated data.
    :return: max_value: The maximum number of targets per publication in the
    annotated data.
    """

    publication_count = len(annotated_metadata)
    first_publication_value = 0
    for target in annotated_metadata[0][publication_annotation_index]:
        if target in annotated_metadata[0][publication_evaluation_data_index]:
            first_publication_value += 1
    min_value = first_publication_value
    max_value = first_publication_value
    value_sum = 0

    for publication in annotated_metadata:
        value = 0
        for target in publication[publication_annotation_index]:
            if target in publication[publication_evaluation_data_index]:
                value += 1
        if value > max_value:
            max_value = value
        if value < min_value:
            min_value = value
        value_sum += value

    avg_value = value_sum / publication_count

    return avg_value, min_value, max_value
