from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer
from manim import *
# Import modules here


class BasicScene(ThreeDScene):
	def construct(self):

		titulo = Title('MLP Caracteres')
		self.play(Write(titulo), run_time=2)


		# Make the neural network
		nn = NeuralNetwork([
			FeedForwardLayer(num_nodes=7),
			FeedForwardLayer(num_nodes=5, activation_function='Sigmoid'),
			FeedForwardLayer(num_nodes=3, activation_function='Sigmoid')
		])
		self.add(nn)

		text1 = Text("Camada de entrada: 120").next_to(nn.input_layers[-1], LEFT).scale(0.3).shift(LEFT)
		self.play(Write(text1), run_time=0.5)

		text2 = Text("Camada escondida: 60").next_to(text1, DOWN).scale(0.3)
		self.play(Write(text2), run_time=0.5)

		text3 = Text("Camada de saída: 26").next_to(text2, DOWN).scale(0.3)
		self.play(Write(text3), run_time=0.5)

		a_in_text = Text("A").scale(0.5).next_to(nn.input_layers[0], LEFT)
		b_in_text = Text("B").scale(0.5).next_to(nn.input_layers[0], LEFT)
		c_in_text = Text("C").scale(0.5).next_to(nn.input_layers[0], LEFT)
		self.play(Write(a_in_text))
		# Make the animation
		forward_pass_animation = nn.make_forward_pass_animation()
		# Play the animation
		self.play(forward_pass_animation, run_time=2)

		# AAAAAAA
		output_matrix = Matrix([[1], [0], [0]]).scale(0.3).move_to(nn.input_layers[-1], aligned_edge=DOWN).shift(RIGHT)
		self.play(Write(output_matrix))
		a_text = Text("A").scale(0.5).next_to(output_matrix, RIGHT)
		self.play(Write(a_text))
		self.wait(1)

		self.remove(output_matrix)
		self.remove(a_text)
		self.remove(a_in_text)

		# BBBBB
		self.play(Write(b_in_text))
		self.play(nn.make_forward_pass_animation(), run_time=2)
		output_matrix = Matrix([[0], [1], [0]]).scale(0.3).move_to(nn.input_layers[-1], aligned_edge=DOWN).shift(RIGHT)
		self.play(Write(output_matrix))
		b_text = Text("B").scale(0.5).next_to(output_matrix, RIGHT)
		self.play(Write(b_text))

		self.wait(1)

		self.remove(output_matrix)
		self.remove(b_text)
		self.remove(b_in_text)

		# CCCCC
		self.play(Write(c_in_text))
		self.play(nn.make_forward_pass_animation(), run_time=2)
		output_matrix = Matrix([[0], [0], [1]]).scale(0.3).move_to(nn.input_layers[-1], aligned_edge=DOWN).shift(RIGHT)

		self.play(Write(output_matrix))
		c_text = Text("C").scale(0.5).next_to(output_matrix, RIGHT)
		self.play(Write(c_text))

		self.wait(2)
