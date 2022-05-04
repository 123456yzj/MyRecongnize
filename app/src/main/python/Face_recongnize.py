import matplotlib.pyplot as plt
def changpic(path,x,y,w,h):
    img = Image.open(path)
    # print(img.size)

    img = img.crop((x, y, x + w, y + h))  # 剪裁图 片

    img = img.resize((128, 128))  # 将图像变成128*128大小的
    img.save("black_10.jpg")  # 保存图片

    plt.figure()  # 用plt打开图片
    plt.imshow(img)
    plt.show()
    print(img.size)
    return img
def detecface(path):
    img = cv2.imread(path)
    color = (0, 255, 0)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img.resize((128, 128))  # 将图像变成128*128大小的
    # img.save("./hsv/new.jpg")  # 保存图片
    # plt.figure()  # 用plt打开图片
    # plt.imshow(grey)
    # plt.show()
    classfier = cv2.CascadeClassifier("D:\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml")

    faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(faceRects) > 0:  # 大于0则检测到人脸
        # print("进入")
        for faceRect in faceRects:  # 单独框出每一张人脸

            x, y, w, h = faceRect
            return changpic(path,x,y,w,h)
            #cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)  # 5控制绿色框的粗细
# x=input("请输入地址：")
# detecface(x)


import cv2
from PIL import Image
import numpy as np
import joblib

label_map = {1:'white',
             2:'black',
             3:'red',
             4:'yellow',
             5: 'cyan',
             6: 'normal'}
#训练集图片的位置
train_image_path ="E:\\the_face_recongnise\face_database\black\black_1.jpg"

size = 128
model_path = 'E:\the_face_recongnise\w_model\\'#修改为绝对路径
#获取HSV颜色特征
hlist = [20, 40, 75, 155, 190, 270, 290, 316, 360]
svlist = [21, 178, 255]
def quantilize(h, s, v):
    '''hsv直方图量化'''
    # value : [21, 144, 23] h, s, v
    h = h * 2
    for i in range(len(hlist)):
        if h <= hlist[i]:
            h = i % 8
            break
    for i in range(len(svlist)):
        if s <= svlist[i]:
            s = i
            break
    for i in range(len(svlist)):
        if v <= svlist[i]:
            v = i
            break
    return 9 * h + 3 * s + v
quantilize_ufunc = np.frompyfunc(quantilize, 3, 1) # 自定义ufunc函数，即将quantilize函数转化为ufunc函数，其输入参数为３个，输出参数为１个。
def colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    nhsv = quantilize_ufunc(hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]).astype(np.uint8) # 由于frompyfunc函数返回结果为对象，所以需要转换类型
    hist = cv2.calcHist([nhsv], [0], None, [72], [0,71]) # 40x faster than np.histogram
    hist = hist.reshape(1, hist.shape[0]).astype(np.int32).tolist()[0]

    return hist
#获得图片列表
def get_image_list(filePath):
    return detecface(filePath)

#提取特征并保存
def get_feat(image,size):
    print("调用提取hog特征函数")
    i = 0
    try:
        #如果是灰度图片  把3改为-1

        image = np.reshape(image, (size, size, 3))
        # print("图片数组重塑" + name_list[i])

    except:
        print("出错")
    fd=[]
    i==0
    hsv = colors(image)
    for v in hsv:
        s = np.str_(v)
        fd=np.concatenate((fd,[s]))
        i+=1
    fd = np.concatenate((fd, ['5\n']))
    # print("Test features are extracted and saved.")
    return fd
#变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray
#提取特征
def extra_feat(train_image_path):
    my_image = get_image_list(train_image_path)
    get_feat(my_image, size)
#训练和测试
def train_and_test():
    clf_type = 'LIN_SVM'
    fds = []
    labels = []
    num = 0
    total = 0
    if clf_type == 'LIN_SVM':
        clf = joblib.load(model_path+'model')
        result_list = []
        print("利用训练好的模型进行预测")
        my_image = get_image_list(train_image_path)
        data_test=get_feat(my_image, size)
        # print("开始")
        # print(len(data_test))
        image_name = train_image_path.split('\\')[0].split('.feat')[0]#此处因测试集路径不同，将第一个下标改为了0
        data_test_feat = data_test[:-1].reshape((1,-1)).astype(np.float64)
        result = clf.predict(data_test_feat)
        result_list.append(image_name+' '+label_map[int(result[0])]+'\n')
    for s in result_list:
        print(s)
    write_to_txt(result_list)
def write_to_txt(list):
    with open('w_result.txt', 'w') as f:
        f.writelines(list)
    print('每张图片的识别结果存放在w_result.txt里面')
def runProgram(train_image_path):
    print("2、训练并预测")
    train_and_test(train_image_path) #训练并预测







