import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import imageio
import sys


class StyleTransfer:

    def __init__(self, content_image, style_image):
        self.STYLE_LAYERS = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)]

        tf.reset_default_graph()

        sess = tf.InteractiveSession()
        model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

        sess.run(model['input'].assign(content_image))
        out = model['conv4_2']

        a_C = sess.run(out)
        a_G = out

        J_content = self.compute_content_cost(a_C, a_G)

        sess.run(model['input'].assign(style_image))

        self.model = model
        self.sess = sess

        J_style = self.compute_style_cost(model, self.STYLE_LAYERS)
        J = self.total_cost(J_content, J_style)

        optimizer = tf.train.AdamOptimizer(2.0)
        self.train_step = optimizer.minimize(J)

        self.J_content = J_content
        self.J_style = J_style
        self.J = J

    @staticmethod
    def compute_content_cost(a_C, a_G):
        """
        Computes the content cost

        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

        Returns:
        J_content -- scalar that you compute using equation 1 above.
        """

        ### START CODE HERE ###
        # Retrieve dimensions from a_G (≈1 line)
        a_G_shapes = a_G.get_shape().as_list()

        # Reshape a_C and a_G (≈2 lines)
        a_C_reshaped = tf.reshape(a_C, [1, a_G_shapes[1] * a_G_shapes[2], a_G_shapes[3]])
        a_G_reshaped = tf.reshape(a_G, [1, a_G_shapes[1] * a_G_shapes[2], a_G_shapes[3]])

        # compute the cost with tensorflow (≈1 line)
        coeff = 1 / (4 * a_G_shapes[1] * a_G_shapes[2] * a_G_shapes[3])
        J_content = coeff * tf.math.reduce_sum(tf.math.square(tf.math.subtract(a_C_reshaped, a_G_reshaped)))
        ### END CODE HERE ###

        return J_content

    @staticmethod
    def gram_matrix(A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)

        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """

        ### START CODE HERE ### (≈1 line)
        GA = tf.linalg.matmul(A, A, transpose_b=True)
        ### END CODE HERE ###

        return GA

    def compute_layer_style_cost(self, a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

        Returns:
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        ### START CODE HERE ###
        # Retrieve dimensions from a_G (≈1 line)
        a_G_shapes = a_G.get_shape().as_list()

        # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
        a_G_reshaped = tf.transpose(tf.reshape(a_G, [a_G_shapes[1] * a_G_shapes[2], a_G_shapes[3]]))
        a_S_reshaped = tf.transpose(tf.reshape(a_S, [a_G_shapes[1] * a_G_shapes[2], a_G_shapes[3]]))
        # Computing gram_matrices for both images S and G (≈2 lines)
        S_gram = self.gram_matrix(a_S_reshaped)
        G_gram = self.gram_matrix(a_G_reshaped)

        # Computing the loss (≈1 line)
        coeff = 1 / (4 * a_G_shapes[3] ** 2 * (a_G_shapes[1] * a_G_shapes[2]) ** 2)
        J_style_layer = coeff * tf.math.reduce_sum(tf.math.square(S_gram - G_gram))

        ### END CODE HERE ###

        return J_style_layer

    def compute_style_cost(self, model, STYLE_LAYERS):
        """
        Computes the overall style cost from several chosen layers

        Arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them

        Returns:
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        # initialize the overall style cost
        J_style = 0

        for layer_name, coeff in STYLE_LAYERS:
            # Select the output tensor of the currently selected layer
            out = model[layer_name]

            # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
            a_S = self.sess.run(out)

            # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
            # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
            # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
            a_G = out

            # Compute style_cost for the current layer
            J_style_layer = self.compute_layer_style_cost(a_S, a_G)

            # Add coeff * J_style_layer of this layer to overall style cost
            J_style += coeff * J_style_layer

        return J_style

    @staticmethod
    def total_cost(J_content, J_style, alpha=10, beta=40):
        """
        Computes the total cost function

        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost

        Returns:
        J -- total cost as defined by the formula above.
        """

        ### START CODE HERE ### (≈1 line)
        J = alpha * J_content + beta * J_style
        ### END CODE HERE ###

        return J

    def model_nn(self, input_image, num_iterations=200):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.model['input'].assign(input_image))

        generated_image = input_image
        for i in range(num_iterations):

            self.sess.run(self.train_step)
            generated_image = self.sess.run(self.model['input'])

            if i % 20 == 0:
                Jt, Jc, Js = self.sess.run([self.J, self.J_content, self.J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))

        return generated_image


def main():
    content_image = imageio.imread(sys.argv[1])
    style_image = imageio.imread(sys.argv[2])
    generated_image_path = sys.argv[3]

    content_image = reshape_and_normalize_image(content_image)
    style_image = reshape_and_normalize_image(style_image)
    generated_image = generate_noise_image(content_image)

    style_transformer = StyleTransfer(content_image, style_image)

    generated_image = style_transformer.model_nn(input_image=generated_image)

    save_image(generated_image_path, generated_image)


main()
