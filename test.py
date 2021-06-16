import imageio
from moviepy.editor import *

reader = imageio.get_reader('test.mp4')
fps = reader.get_meta_data()['fps']
duration = reader.get_meta_data()['duration']
print(reader.get_meta_data())
kargs = { 'macro_block_size': None }
writer = imageio.get_writer('test1.mp4', fps=fps, **kargs)


for im in reader:
    writer.append_data(im[:, :, :])
writer.close()

videoclip1 = VideoFileClip("test.mp4")
videoclip2 = VideoFileClip("test1.mp4")

videoclip2.audio = videoclip1.audio
videoclip2.write_videofile("test1.mp4", verbose=False, logger=None)
