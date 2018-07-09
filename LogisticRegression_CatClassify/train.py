import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import functions as fun

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 预处理
# train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3)
m_train = train_set_x_orig.shape[0]  # 209
m_test = test_set_x_orig.shape[0]  # 50
num_px = train_set_x_orig.shape[1]  # 64

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T  # train_set_x_flatten shape: (12288, 209)
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T  # test_set_x_flatten shape: (12288, 50)

# standardize dataset
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.
# 预处理结束


# # 开始训练
# learning_rates = [0.01, 0.001, 0.0001]
# models = {}
# for i in learning_rates:
#     print("learning rate is: " + str(i))
#     models[i] = fun.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
#                           print_cost=False)
#     print('\n' + "-------------------------------------------------------" + '\n')
# # 训练结束
#
#
# # 数据分析
# # 不同学习速率下的cost
# for i in learning_rates:
#     plt.plot(np.squeeze(models[i]["costs"]), label=str(models[i]["learning_rate"]))
#
# plt.ylabel('cost')
# plt.xlabel('iterations')
#
# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()

# 识别自己的图片
d = fun.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=True)

fname = "cat1.jpg"
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = fun.predict(d["w"], d["b"], my_image)

str = "y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
    int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture."
plt.imshow(my_image.reshape(64, 64, 3))
plt.title(str,color='red',fontsize='17')
plt.show()
