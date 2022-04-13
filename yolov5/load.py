import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
model = torch.hub.load('', 'custom', path='artifacts/run/best.pt', source='local') # local repo




for i in range(1,2):
	img = f'jordan/jordanTestBatch1 {str(i)}.jpg'
	results = model(img)
	results.print()
	cv2.imshow('image', np.squeeze(results.render()))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
