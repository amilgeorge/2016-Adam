import imageio

images = []
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave('/path/to/movie.gif', images)