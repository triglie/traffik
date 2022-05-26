import sys
import numpy as np
import moviepy.editor

class ExtractVideoDuration:
    @staticmethod
    def get_duration(path:str) -> int:
        try:
            return int(moviepy.editor.VideoFileClip(path).duration*1000)
        except ValueError:
            print("ERROR: file not found.")
            exit(-1)
        return None


def extract_action_frames(path):
    """ Ritorna la lista degli indici dei frame in cui 
        compare un oggetto. 
    """
    with open(path, 'r') as csvfile:
        lines = csvfile.readlines()[3:]
        action_frames = []
        for line in lines:
            frame = line.split(',')[0]
            action_frames.append(int(frame)/2)
        return action_frames


def build_frame_binary_array(nframes, action_frames):
    """ Crea un array binario in cui l'i-esimo bit 
        è 1 se accade qualcosa (il frame è un action frame), 
        altrimenti è 0 
    """
    return [ 1 if i in action_frames else 0 for i in range(nframes) ]


if __name__ == '__main__': 

    if len(sys.argv) != 3:
        print("error, usage: python generate_user_sum.py <video.mp4> <file_with_summaries.csv>")
        sys.exit(-1)

    _, video, path = sys.argv 
    
    nframes:int = ExtractVideoDuration.get_duration(video)/60
    user_summ = build_frame_binary_array(nframes, extract_action_frames(path))
    print(user_summ)