import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# lfw_people = fetch_lfw_people(data_home='./face', slice_=(slice(70, 195), slice(78, 172)))
lfw_people = fetch_lfw_people(data_home='./face', slice_=(slice(90, 170), slice(85, 165)), resize=0.2375) # 19 x 19 px

print(lfw_people.images.shape)

images_num = lfw_people.images.shape[0]
img_h = lfw_people.images[0].shape[0]
img_w = lfw_people.images[0].shape[1]

out_images = lfw_people.images.reshape(images_num, img_h * img_w).T

np.save('./face_images', out_images)

print(out_images.shape)
img = out_images.T[0].reshape(img_h, img_w)
print(img.shape)

plt.imshow(img)
plt.show()

# img = lfw_people.images[2]
# plt.imshow(img)
# plt.show()
