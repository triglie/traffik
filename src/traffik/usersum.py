import moviepy.editor


def get_duration(path:str) -> int:
    """ Get duration of a video in seconds """
    try:
        return int(moviepy.editor.VideoFileClip(path).duration*1000)
    except ValueError:
        print("ERROR: file not found.")
        exit(-1)


def extract_action_frames(path):
    """ Returns a list containing the index of all the 
        frames containing an object. 
    """
    with open(path, 'r') as csvfile:
        lines = csvfile.readlines()[3:]
        action_frames = []
        for line in lines:
            frame = line.split(',')[0]
            action_frames.append(int(frame)/2)
        return action_frames


def build_frame_binary_array(nframes, action_frames):
    """ Returns a binary array where the i-th bit is 1 if something
        is happening in the scene (action frame), 0 otherwise.
    """
    return [ 1 if i in action_frames else 0 for i in range(nframes) ]