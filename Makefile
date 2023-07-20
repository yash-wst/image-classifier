# Trains the model with available training data
train:
	python3 -m pip install tensorflow
	bash train.sh

# Creates a docker image with the model
bin:
	docker build -t image_classifier .

# Fetches the training data and extracts it
fetchimages:
	python3 create_training_data.py