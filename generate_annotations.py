import os
import pandas as pd
import argparse

from utils import non_max_suppression
import config


def generate_annotations(model_output_dir, model_output_list, annotations_output_dir, output_file_insertion):

    annotation_files = dict()

    for output in model_output_list:

        index_before_number = output.rfind('_')
        annotation_filename = output[:index_before_number]

        if annotation_filename not in annotation_files:
            annotation_files[annotation_filename] = []

        output_name = os.path.join(output_dir, model_output_list)

        with open(output_name, "r") as f:
            lines = [line.strip('\n').split(' ') for line in f.readlines()]

        annotation_files[annotation_filename] += lines

    columns = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Delta Time (s)", "Annotation"]

    for annotation_filename in annotation_files.keys():

        raven_annotations = []
        string_annotations = annotation_files[annotation_filename]

        annotations = [
            [int(annotation[0]), float(annotation[1]), float(annotation[2]), float(annotation[3])]
                 for annotation in string_annotations]

        # perform non maximal suppression
        annotations = non_max_suppression(annotations)

        # format the annotations in preparation for turning them into Raven rows
        # convert x_center and width format to begin, end, width
        # and lookup the class labels
        # the first part needs to be done now (before the Raven loop) because
        # we want to sort by starting time first

        classes = config.CLASSES

        for i in range(len(annotations)):
            current_annotation = annotations[i]
            class_label, _, x_center, width = current_annotation

            x_begin = x_center - width/2
            x_end = x_begin+width

            annotations[i] = [classes[class_label], x_begin, x_end, width]

        # sort by start time
        annotations.sort(key=lambda x: x[1])

        for row_num, annotation in enumerate(annotations):
            raven_row = [f"{row_num+1}", "Spectrogram 1", "1", annotation[1], annotation[2], annotation[3], annotation[0]]
            raven_annotations.append(raven_row)

        df = pd.DataFrame(raven_annotations, columns=columns)
        df.reset_index(drop=True, inplace=True)

        output_filename = annotation_filename + output_file_insertion + '_annotations.txt'
        df.to_csv(os.path.join(annotations_output_dir, output_filename), sep='\t', index=False)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output-dir", help="Directory with output files from the model" default="outputs")
    parser.add_argument("--model-output-list", help="If provided the script will use only spectrograms from this text file to make the annotations file")
    parser.add_argument("--annotations-output-dir", help="If provided the script will output the annotations file to this directory")
    args = parser.parse_args()

    model_output_dir = args.model_output_dir

    if args.model_output_list == None:
        model_output_list = os.listdir(model_output_dir)
        output_file_insertion = ""
    else:
        with open(args.model_output_list, "r") as f:
            model_output_list = [line.strip()[:-2] + 'txt' for line in f.readlines()]
        output_file_insertion = '_' + os.path.basename(args.model_output_list[:-4])

    if args.annotations_output_dir == None:
        annotations_output_dir = "."
    else:
        annotations_output_dir = args.annotations_output_dir

    generate_annotations(model_output_dir, model_output_list, annotations_output_dir, output_file_insertion)


if __name__ == "__main__":
    main()
