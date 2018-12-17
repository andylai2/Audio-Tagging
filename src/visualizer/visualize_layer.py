from torch.optim import Adam
from visualizer.visualize_util import *

class CNNLayerVisualization():
	"""Produces an image of the optimal filter for as specific layer"""

	def __init__(self, model, selected_layer, selected_filter):
		self.model = model
		self.model.eval()
		self.selected_layer = selected_layer
		self.selected_filter = selected_filter
		self.conv_output = 0

		# Random image
		self.created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))

		if not os.path.exists('../filter_outputs'):
			os.makedirs('../filter_outputs')

	def hook_layer(self):
		def hook_function(module, grad_in, grad_out):
			# Gets the conv output of the selected filter (from selected layer)
			self.conv_output = grad_out[0, self.selected_filter]

		# Hook the selected layer
		self.model[self.selected_layer].register_forward_hook(hook_function)

	def visualise_layer_with_hooks(self):
		self.hook_layer()
		self.processed_image = preprocess_image(self.created_image, False)

		optimizer = Adam([self.processed_image], lr = 0.1, weight_decay=1e-6)
		for i in range(1,31):
			optimizer.zero_grad()

			x = self.processed_image
			for index, layer in enumerate(self.model):
				x = layer(x)
				if index == self.selected_layer:
					# forward hook function
					break

			loss = -torch.mean(self.conv_output)
			loss.backward()
			optimizer.step()
			self.created_image = recreate_image(self.processed_image)

			if i % 5 == 0:
				im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
						  '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
				save_image(self.created_image, im_path)
