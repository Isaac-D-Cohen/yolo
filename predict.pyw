import tkinter as tk
from tkinter import filedialog

from utils import ensure_dir_exists
from make_spectrograms import make_spectrograms
from predict import predict
from generate_annotations import generate_annotations

# create textbox/entry objects
entry_select_chkpt = None
texbx_select_audios = None
entry_select_output_dir = None


def select_checkpoint():
    cf = filedialog.askopenfilename(filetypes=[('Pytorch Checkpoints', '*.pth.tar')])
    if cf:
        entry_select_chkpt.delete(0, "end")
        entry_select_chkpt.insert(0, cf)

def select_audios():
    af = filedialog.askopenfilenames(filetypes=[('WAV files', '*.wav')])
    if af:
        s = "\n".join(af)
        texbx_select_audios.delete("1.0", "end")
        texbx_select_audios.insert("1.0", s)

def select_output_dir():
    od = filedialog.askdirectory()
    if od:
        entry_select_output_dir.delete(0, "end")
        entry_select_output_dir.insert(0, od)

def run():

    checkpoint_filename = entry_select_chkpt.get()
    audio_filenames = texbx_select_audios.get("1.0", "end-1c").split('\n')
    output_dir = entry_select_output_dir.get()

    if len(checkpoint_filename) == 0 or len(audio_filenames) == 0 or len(output_dir) == 0:
        tk.messagebox.showerror("Error", "To proceed, please fill out all fields. Aborting.")
        return

    # make the directories
    inference_input = config.INFERENCE_IMAGES
    inference_output = config.INFERENCE_OUTPUTS

    os.makedirs(inference_input, exist_ok=True)
    os.makedirs(inference_output, exist_ok=True)

    # ok, step 1: make the spectrograms!

    clip_len = config.CLIP_LEN
    overlap = config.OVERLAP
    step = clip_len-overlap

    for audio_file in audio_filenames:

        spectrograms = make_spectrograms(audio_file, clip_len, step)

        # remove the .wav
        sound_name = audio_file[:-4]

        # save the tensor
        filename = os.path.join(inference_input, sound_name + f"_{i}.pt")
        torch.save(spectrogram, filename)

    # step 2: make the predictions
    checkpoint_filename = checkpoint_filename[:-8]
    checkpoint_path, checkpoint_name = os.path.split(checkpoint_filename)

    predict(checkpoint_name, checkpoint_path)

    # step 3: generate the output files
    output_list = os.listdir(inference_output)
    output_file_insertion = ""

    generate_annotations(inference_output, output_list, output_dir, output_file_insertion)

    tk.messagebox.showinfo("Done!", "All done!")


def main():

    root = tk.Tk()

    # set window properties
    root.title("Run Network")
    root.configure(background="white")
    root.minsize(500, 300)

    # setup rows and columns
    root.columnconfigure([0,3], weight=0, minsize=75)
    root.columnconfigure(1, weight=1, minsize=150)
    root.columnconfigure(2, weight=0, minsize=150)
    root.rowconfigure([0,1,2,3,4,5,6,7,8], weight=1, minsize=20)


    # create entries and buttons
    global entry_select_chkpt
    global texbx_select_audios
    global entry_select_output_dir

    entry_select_chkpt=tk.Entry(master=root)
    entry_select_chkpt.grid(row=1, column=1, sticky="news")

    button_select_chkpt=tk.Button(master=root, text="Select Checkpoint", command=select_checkpoint)
    button_select_chkpt.grid(row=1, column=2, sticky="news")

    texbx_select_audios=tk.Text(master=root, height=4)
    texbx_select_audios.grid(row=3, column=1, sticky="news")

    button_select_audios=tk.Button(master=root, text="Select Audio Files", command=select_audios)
    button_select_audios.grid(row=3, column=2, sticky="news")

    entry_select_output_dir=tk.Entry(master=root)
    entry_select_output_dir.grid(row=5, column=1, sticky="news")

    button_select_output_dir=tk.Button(master=root, text="Select Output Folder", command=select_output_dir)
    button_select_output_dir.grid(row=5, column=2, sticky="news")

    button_run=tk.Button(master=root, text="Run", command=run)
    button_run.grid(row=7, column=1, columnspan=2, sticky="news")


    root.mainloop()


if __name__ == "__main__":
    main()
