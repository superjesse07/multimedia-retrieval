import os
import shutil

def process_obj_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        if files:
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            query_file = files[0]  # Selecting the first file as query
            rest_files = files[1:]  # The rest files are database

            # Process query file
            input_filepath = os.path.join(root, query_file)
            query_output_dir = os.path.join(output_subdir, "query")
            if not os.path.exists(query_output_dir):
                os.makedirs(query_output_dir)
            output_filepath = os.path.join(query_output_dir, query_file)
            shutil.copy(input_filepath, output_filepath)

            # Process the rest files
            database_output_dir = os.path.join(output_subdir, "samples")
            if not os.path.exists(database_output_dir):
                os.makedirs(database_output_dir)
            for file in rest_files:
                input_filepath = os.path.join(root, file)
                output_filepath = os.path.join(database_output_dir, file)
                shutil.copy(input_filepath, output_filepath)

input_directory = 'normalised_v2_dataset'
output_directory = 'query_dataset'

process_obj_files(input_directory, output_directory)

print("Separation process completed.")
