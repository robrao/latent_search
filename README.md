# Latent Search
The goal of this repo is to test the idea of using a latent space representation of an input
to search for similar inputs. This test is conducted with the MNIST data set, and the latent
space is created by using an autoencoder (type?). If two images are encoded, and then we
take a dot product of the two latent representations, it should give us a measure of the
similarity of the images. We can test that by simply visualizing the images we find a high
measure on. If this does work as described then we would have a method to search for simliar
inputs by simply using the latent space representations.
