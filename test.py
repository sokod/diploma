import imageio

reader = imageio.get_reader('test.mp4')
fps = reader.get_meta_data()['fps']
duration = reader.get_meta_data()['duration']
change_img = imageio.imread('download.jpg')
print(reader.get_meta_data())

writer = imageio.get_writer('test_1.mp4', fps=fps)

step = 1
total_fps = fps * duration
for im in reader:
    writer.append_data(change_img)
    progress = (step / total_fps) * 100
    print(progress)
    step+=1
writer.close()
