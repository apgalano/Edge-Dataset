# Edge-Dataset
Latency and confidence measurements obtained from an edge-assisted object recognition system.

The records are obtained by tuning the image encoding rate and Neural Network input layer size and measuring the latency on completing each system operation, i.e. encoding the image, transmiting it wirelessly, decoding and rotating at the server, and performing object recognition with YOLO on the server's GPU. Moreover, we document the achievable frame rate as a result of the total latency, as well as the object recognition confidence and cumulative confidence for all identified objects of each image.

The Edge-Dataset.pdf file contains a detailed description of the system and the obtained measuremnts.
The process_data.py file contains a small script for visualising the dataset.
