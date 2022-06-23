"""
Simple GUI
"""
import datetime
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
from predict import predict

FPS = 25
SEQ_TAB = [(0, '<loading..>')]


def predict_for_frame(frames):

    preq = SEQ_TAB[0]
    for i in range(1, len(SEQ_TAB)):
        if SEQ_TAB[i][0] < frames:
            preq = SEQ_TAB[i]
        else:
            break
    return preq[1]


def update_duration(event):
    """ updates the duration after finding the duration """
    end_time["text"] = str(datetime.timedelta(seconds=vid_player.duration()))
    progress_slider["to"] = vid_player.duration() * FPS


def update_scale(event):
    """ updates the scale value """
    progress_slider.set(vid_player.current_duration() * FPS)
    prediction_label['text'] = 'Prediction: ' + \
        predict_for_frame(vid_player.current_duration() * FPS)


def load_video():
    """ loads the video """
    file_path = filedialog.askopenfilename(filetypes=[("Video", ".avi")])
    global SEQ_TAB
    if file_path:
        vid_player.load(file_path)

        progress_slider.config(to=0, from_=0)
        progress_slider.set(0)
        play_pause_btn["text"] = "Play"
        load_vid_label['text'] = file_path
        load_vid_label['anchor'] = 'w'
        if load_net_label['text'] != '<Wybierz plik>':
            SEQ_TAB = predict(load_net_label['text'], load_vid_label['text'])


def load_net():
    """ loads the video """
    global SEQ_TAB
    file_path = filedialog.askopenfilename(filetypes=[("Net file", ".hdf5")])

    if file_path:
        # LOAD NET

        load_net_label['text'] = file_path
        load_net_label['anchor'] = 'w'
        SEQ_TAB = predict(load_net_label['text'], load_vid_label['text'])


def seek(value):
    """ used to seek a specific timeframe """
    vid_player.seek(int(value) // FPS)


def skip(value: int):
    """ skip seconds """
    vid_player.skip_sec(value)
    progress_slider.set(progress_slider.get() + value)


def play_pause():
    """ pauses and plays """
    if vid_player.is_paused():
        vid_player.play()
        play_pause_btn["text"] = "Pause"

    else:
        vid_player.pause()
        play_pause_btn["text"] = "Play"


def video_ended(event):
    """ handle video ended """
    progress_slider.set(progress_slider["to"])
    play_pause_btn["text"] = "Play"


root = tk.Tk()
root.title("Analiza czynności osób")
root.geometry("720x480")

topsection = tk.Frame(root, borderwidth=2, bg='#BDD0e7')
topsection.pack(fill=tk.BOTH)

load_vid_section = tk.Frame(topsection, borderwidth=2, bg='#BDD0e7')
load_vid_section.pack(fill=tk.BOTH)

load_vid_btn = tk.Button(load_vid_section, text="Load video", command=load_video, width=8)
load_vid_btn.pack(side=tk.LEFT)
load_vid_label = tk.Label(load_vid_section, text='<Wybierz plik>', justify='left')
load_vid_label.pack(fill=tk.BOTH)

load_net_section = tk.Frame(topsection, borderwidth=2, bg='#BDD0e7')
load_net_section.pack(fill=tk.BOTH)

load_net_btn = tk.Button(load_net_section, text="Load net", command=load_net, width=8)
load_net_btn.pack(side=tk.LEFT)
load_net_label = tk.Label(load_net_section, text='<Wybierz plik>', justify='left')
load_net_label.pack(fill=tk.BOTH)

vid_player = TkinterVideo(scaled=False, pre_load=False, master=root)
vid_player.pack(expand=False, fill="both")

bottom_section = tk.Frame(root, borderwidth=2)
bottom_section.pack(fill=tk.BOTH, side='bottom')

predplay_section = tk.Frame(bottom_section, borderwidth=2)
predplay_section.pack(fill=tk.BOTH, side='bottom')

play_pause_btn = tk.Button(predplay_section, text="Play", command=play_pause, width=8)
play_pause_btn.pack(side=tk.LEFT)

prediction_label = tk.Label(predplay_section, text='<Prediction>', font=("Arial", 16))
prediction_label.pack(fill=tk.BOTH)

start_time = tk.Label(bottom_section, text=str(datetime.timedelta(seconds=0)))
start_time.pack(side="left")

progress_slider = tk.Scale(bottom_section, from_=0, to=0, orient="horizontal", command=seek)
progress_slider.pack(side="left", fill="x", expand=True)

end_time = tk.Label(bottom_section, text=str(datetime.timedelta(seconds=0)))
end_time.pack(side="left")

vid_player.bind("<<Duration>>", update_duration)
vid_player.bind("<<SecondChanged>>", update_scale)
vid_player.bind("<<Ended>>", video_ended)

root.mainloop()
