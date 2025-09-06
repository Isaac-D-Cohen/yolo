from sys import argv
import pandas as pd
import os
import config

def main():

    clip_len = config.CLIP_LEN
    overlap = config.OVERLAP

    if len(argv) < 2:
        print(f"Format: {argv[0]}\t<file_with_spectrograms_to_include>")
        exit(0)

    spectrograms = dict()
    with open(argv[1], "r") as f:
        for line in f.readlines():
            # remove the \n from the end and the .pt
            spec_name = line.strip('\n')[:-3]
            # isolate the spectrogram name and number
            spec_number = spec_name.split('_')[-1]
            audio_name = spec_name.removesuffix("_" + spec_number)
            spec_number = int(spec_number)

            if audio_name in spectrograms:
                spectrograms[audio_name].append(spec_number)
            else:
                spectrograms[audio_name] = [spec_number]

    filter_filename = os.path.basename(argv[1][:-4])

    columns = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Delta Time (s)", "Annotation"]

    for sound_filename in spectrograms.keys():

        output_df = pd.DataFrame(list(), columns=columns)

        all_annotations = pd.read_csv(sound_filename + '.txt', sep='\t')

        spectrograms[sound_filename].sort()

        for spec_number in spectrograms[sound_filename]:

            # find the window
            start = (clip_len-overlap)*spec_number
            end = start+clip_len

            # find all annotations within this window
            start_before_end_of_window = all_annotations['End Time (s)'] > start
            end_after_start_of_window = all_annotations['Begin Time (s)'] < end
            subset_in_window = all_annotations[start_before_end_of_window & end_after_start_of_window]

            # now truncate the ones that go over the edges of the window
            truncate_at_start = subset_in_window['Begin Time (s)'] > start
            subset_in_window[truncate_at_start]['Begin Time (s)'] = start
            truncate_at_end = subset_in_window['End Time (s)'] < end
            subset_in_window[truncate_at_end]['End Time (s)'] = end

            if output_df.empty:
                output_df = subset_in_window
            else:
                output_df = pd.concat([output_df, subset_in_window])

        output_df.reset_index(drop=True, inplace=True)
        output_df.to_csv(sound_filename + '_' + filter_filename + '_annotations.txt', sep='\t', index=False)






if __name__ == "__main__":
    main()
