'''
@author linrenkang
神经网络预测
'''
import network

class Model:
    net = None

    # 私有函数
    def __init__(self):
        pass

    # 推理预测
    def Predict(self, trainFile):
        model = network.Network()
        model.combined.load_weights(trainFile)
        model.sample_images("predict")
'''
    基于 keras 调用 
'''
if __name__ == "__main__":
    trainFile = "./logs/gan-1.h5"

    myModel = Model()
    myModel.Predict(trainFile)
