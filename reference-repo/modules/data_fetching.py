import os
from datetime import datetime
from typing import Union
# import gzip

import isal
from isal import igzip_threaded
# from pgzip import pgzip
# import zlib_ng
# from zlib_ng import gzip_ng_threaded#


def extract_gzip(gzip_filename: str, read_mode='rb',
                 threads=os.cpu_count() - 1) -> Union[str, bytes, None]:
    """
    Extracts a .gz file using the given parameters.

    :param gzip_filename: Filename of the `.gz` archive
    :param read_mode: Read mode (see the isal/gzip documentation for more info)
    :param threads: Number of threads to use. Please beware: Even when setting
    a thread number, this might have no effect.
    :return: Content of the `.gz` file
    """
    start_time = datetime.now()
    print(f"Extracting the data of the .gz archive using isal (start time: "
          f"{start_time}).")
    # all variants below configured for {os.cpu_count() - 1} threads use less
    # very often
    with isal.igzip_threaded.open(gzip_filename, threads=threads,
                                  mode=read_mode) as gz_file:
        # alternatives below, if isal becomes unsupported; gzip takes about 2x
        # the time as the other alternatives
        # with gzip.open(gzip_filename, 'rb') as gz_file:
        # with zlib_ng.gzip_ng_threaded.open(gzip_filename,
        #                                    threads=os.cpu_count() - 1)\
        #    as gz_file:
        # with pgzip.open(gzip_filename, 'rb', thread=os.cpu_count() - 1)\
        #    as gz_file:
        extracted_file = gz_file.read()
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Extraction finished (end time: {end_time}).")
    print(f"Duration of the extraction: {duration}.")
    return extracted_file
