import os
import pandas as pd

output_dir = "outputs"
confidence_threshold = .4

def main():
    annotation_files = dict()
    output_list = os.listdir(output_dir)

    for output in output_list:

        index_before_number = output.rfind('_')
        annotation_filename = output[:index_before_number]

        if annotation_filename not in annotation_files:
            annotation_files[annotation_filename] = []

        output_name = os.path.join(output_dir, output)

        with open(output_name, "r") as f:
            lines = [line.strip('\n').split(' ') for line in f.readlines()]

        annotation_files[annotation_filename] += lines

    columns = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Delta Time (s)", "Annotation"]

    for annotation_filename in annotation_files.keys():

        raven_annotations = []
        unprocessed_annotations = annotation_files[annotation_filename]
        annotations = []

        # go through the list of annotations and convert to ints and floats and filter by confidence
        for annotation in unprocessed_annotations:
            if float(annotation[1]) >= confidence_threshold:
                cls = "call" if annotation[0] == "0" else "song"
                annotation = [cls, float(annotation[2]), float(annotation[3]), float(annotation[4])]
                annotations.append(annotation)

        # sort by start time
        annotations.sort(key=lambda x: x[1])

        for row_num, annotation in enumerate(annotations):
            raven_row = [f"{row_num+1}", "Spectrogram 1", "1", annotation[1], annotation[2], annotation[3], annotation[0]]
            raven_annotations.append(raven_row)

        df = pd.DataFrame(raven_annotations, columns=columns)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(annotation_filename + '_annotations.txt', sep='\t', index=False)


if __name__ == "__main__":
    main()
