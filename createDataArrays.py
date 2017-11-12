from PIL import Image
import numpy as np
import tensorflow as tf
import io
import random

LABEL_DICT = {"POSITION-RIGHT\n":0,"POSITION-LEFT\n":1,"POSITION-SUPINE\n":2,"POSITION-PRONE\n":3,"POSITION-UPRIGHT\n":4}
#NUMBER_EXAMPLES = 60000
NUMBER_EXAMPLES = 6000
#NUMBER_EXAMPLES = 50000
#TARGET_SAVE_FILE = "D:\\tmp\\sleep_train_data_60k.npz"
TARGET_SAVE_FILE = "D:\\tmp\\sleep_train_data_6k.npz"
#TARGET_SAVE_FILE = "D:\\tmp\\sleep_test_data_50k.npz"
TARGET_IMAGE_SIZE = [128,106]

def read_data_file(path):
	data_file = open(path,"r")
	labels= list()
	imagePaths = list()
	
	for line in data_file.readlines():
		#print("line:",line)
		#split line on space
		tokens = line.split(" ")
		#print("tokens:",tokens)
		if len(tokens)==2:
			imagePaths.append(tokens[0])
			labels.append(tokens[1])

	return {"ip":imagePaths,"l":labels}


def convert_labels_to_numbers(labels):
	print("converting labels")
	int_labels = []
	for idx in range(0,NUMBER_EXAMPLES):
		int_label = LABEL_DICT.get(labels[idx])
		int_labels.append(int_label)
	return np.array(int_labels)

def convert_image_paths_to_array(image_paths):
	print("reading and stacking images")
	full_array_list = []

	for idx in range(0,NUMBER_EXAMPLES):
		image_path = image_paths[idx]
		if(idx%100==0):
			print(idx," of ", NUMBER_EXAMPLES," done")
		if(len(image_path)>0):
			image = Image.open(image_path)
			#print("image_array.shape:",image_array.shape)
			# resize image to 128x106 	
			# image.thumbnail([128,106])
			image.thumbnail(TARGET_IMAGE_SIZE)
			image_array = np.array(image.getdata())

			# normalize between 0.0 and 1.0
			image_array = image_array + abs(image_array).max()
			image_array = image_array / image_array.max()

			full_array_list.append(image_array)
			#print("full_image_array.shape:",full_image_array.shape)
	print("stacking list of size ",len(full_array_list))	
			
	full_image_array= np.stack(full_array_list)
	return full_image_array




def main():
	
	return_dict = read_data_file("D:\\tmp\\testImages\\resultFile_train.txt")
	#return_dict = read_data_file("D:\\tmp\\testImages\\resultFile_test.txt")
	imagePaths = return_dict["ip"]
	labels = return_dict["l"]
	#print(imagePaths)
	#print(labels)

	#shuffle lists
	combined = list(zip(imagePaths,labels))
	random.shuffle(combined)
	imagePaths[:],labels[:] = zip(*combined)
	# convert everything	
	int_labels = convert_labels_to_numbers(labels)
	all_images = convert_image_paths_to_array(imagePaths)
	print("labels:",int_labels[0:10])
	print("all_images.shape:",all_images.shape)

	#save all
	print("Saving to ",TARGET_SAVE_FILE)
	np.savez(TARGET_SAVE_FILE,image_data = all_images,labels = int_labels)
	print("done saving.")
	print("labels histogram:",np.histogram(int_labels,bins=5))

if __name__ == "__main__":
	
	main()