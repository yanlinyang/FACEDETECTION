# -*- coding: utf-8 -*-

from PyQt5.QtCore import *

from PyQt5.QtWidgets import *

from PyQt5.QtGui import *
import numpy as np
import cv2,os,time,csv
from skimage import io
import pyttsx3
import dlib         # 人脸处理的库 Dlib
from Ui_main import Ui_MainWindow
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
# 返回单张图像的 128D 特征
def return_128d_features(path_img):
    img_rd = io.imread(path_img)
    faces = detector(img_rd, 1)

    print("%-40s %-20s" % ("检测到人脸的图像 / Image with faces detected:", path_img), '\n')

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        print("no face")

    return face_descriptor

engine = pyttsx3.init()
engine.setProperty('voice', engine.getProperty('voices')[2].id)

# 将文件夹中照片特征提取出来, 写入 CSV
def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for i in range(len(photos_list)):
            # 调用return_128d_features()得到128d特征
            print("%-40s %-20s" % ("正在读的人脸图像 / Image to read:", path_faces_personX + "/" + photos_list[i]))

            features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
            #  print(features_128d)
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        print("文件夹内图像文件为空 / Warning: No images in " + path_faces_personX + '/', '\n')

    # 计算 128D 特征的均值
    # personX 的 N 张图像 x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=int, order='C')

    return features_mean_personX

def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1,dtype='float64')
    feature_2 = np.array(feature_2,dtype='float64')
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("dist=",dist)
    return dist

class MainWindow(QMainWindow, Ui_MainWindow):


    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.capture_button.setVisible(False)
        self.capture_end_button.setVisible(False)
        self.timer_camera = QTimer(self)
        self.cap = cv2.VideoCapture(0)
        self.timer_camera.timeout.connect(self.show_Dlib2)

        self.timer_camera.start(10)
        # self.label.setPixmap(QPixmap("../images/111.jpg"))
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.ss_cnt = 0
        self.press_n_flag = 0
        self.press_s_flag = 0
        self.font = cv2.FONT_ITALIC
        self.faces_cnt = 0
        self.save_flag = 0
        self.frame_start_time = 0

        with open("data/person_all.csv", 'r') as f:
            reader = csv.reader(f)
            self.name_known_list = np.array(list(reader))
        with open("data/features_all.csv", 'r') as f:
            reader = csv.reader(f)
            self.features_known_list = list(reader)

        self.newAddName = ''
        self.newAdd_Button.clicked.connect(self.getName)
        self.capture_button.clicked.connect(self.capture)
        self.capture_end_button.clicked.connect(self.capture_end)

        self.feature_extraction_Button.clicked.connect(self.featureExtraction)
        self.face_recognition_Button.clicked.connect(self.faceRecognition)


    def show_pic(self):
        success, frame = self.cap.read()
        if success:
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(showImage))
        self.timer_camera.start(10)

    def show_Dlib2(self):
        flag, img_rd = self.cap.read()        # Get camera video stream
        self.faces = detector(img_rd, 0)         # Use dlib face detector

        # 5. 检测到人脸 / Face detected
        if len(self.faces) != 0:
            # 矩形框 / Show the HOG of faces
            for k, d in enumerate(self.faces):
                # 计算矩形框大小 / Compute the size of rectangle box
                height = (d.bottom() - d.top())
                width = (d.right() - d.left())
                hh = int(height/2)
                ww = int(width/2)

                # 6. 判断人脸矩形框是否超出 480x640
                if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                    cv2.putText(img_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    color_rectangle = (0, 0, 255)
                    self.save_flag = 0
                else:
                    color_rectangle = (255, 255, 255)
                    self.save_flag = 1

                cv2.rectangle(img_rd,
                              tuple([d.left() - ww, d.top() - hh]),
                              tuple([d.right() + ww, d.bottom() + hh]),
                              color_rectangle, 2)

                # 7. 根据人脸大小生成空的图像 / Create blank image according to the shape of face detected
                img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)
                if self.save_flag:
                    # 8. 按下 's' 保存摄像头中的人脸到本地 / Press 's' to save faces into local images
                        # 检查有没有先按'n'新建文件夹 / Check if you have pressed 'n'
                    if self.press_n_flag == 1 and self.press_s_flag == 1:

                        for ii in range(height*2):
                            for jj in range(width*2):
                                img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                        print("写入啊")
                        cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg",
                                        img_blank)
                        print("写入本地 / Save into：",
                                  str(self.current_face_dir) + "/img_face_" + str(self.ss_cnt) + ".jpg")
                        print("写入啊2")
                        self.press_s_flag = 0

                    else:
                        pass
                        # print("请先按 'N' 来建文件夹, 按 'S' / Please press 'N' and press 'S'")
                else:
                    self.tipsShow(self.newAddName + "请调整位置 / Please adjust your position")

            self.faces_cnt = len(self.faces)

        # 9. 生成的窗口添加说明文字 / Add note on cv2 window
        # self.draw_note(img_rd)
        # 10. 按下 'q' 键退出 / Press 'q' to exit
        # if kk == ord('q'):
        #     break
        self.update_fps()
        self.draw_note(img_rd)
        show = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(showImage))
        self.timer_camera.start(10)

    def draw_note(self, img_rd):
        # 添加说明 / Add some statements
        cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)


    def show_Dlib(self,stream):

        while stream.isOpened():
            flag, img_rd = stream.read()        # Get camera video stream
            kk = cv2.waitKey(1)
            faces = detector(img_rd, 0)         # Use dlib face detector
            print(len(faces))

            # 5. 检测到人脸 / Face detected
            if len(faces) != 0:
                # 矩形框 / Show the HOG of faces
                for k, d in enumerate(faces):
                    # 计算矩形框大小 / Compute the size of rectangle box
                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height/2)
                    ww = int(width/2)

                    # 6. 判断人脸矩形框是否超出 480x640
                    if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                        cv2.putText(img_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                        color_rectangle = (0, 0, 255)
                        save_flag = 0
                        if kk == ord('s'):
                            print("请调整位置 / Please adjust your position")
                    else:
                        color_rectangle = (255, 255, 255)
                        save_flag = 1

                    cv2.rectangle(img_rd,
                                  tuple([d.left() - ww, d.top() - hh]),
                                  tuple([d.right() + ww, d.bottom() + hh]),
                                  color_rectangle, 2)

                    # 7. 根据人脸大小生成空的图像 / Create blank image according to the shape of face detected
                    img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                    if save_flag:
                        # 8. 按下 's' 保存摄像头中的人脸到本地 / Press 's' to save faces into local images
                        if kk == ord('s'):
                            # 检查有没有先按'n'新建文件夹 / Check if you have pressed 'n'
                            if self.press_n_flag:
                                self.ss_cnt += 1
                                for ii in range(height*2):
                                    for jj in range(width*2):
                                        img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                                cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                                print("写入本地 / Save into：", str(self.current_face_dir) + "/img_face_" + str(self.ss_cnt) + ".jpg")
                            else:
                                print("请先按 'N' 来建文件夹, 按 'S' / Please press 'N' and press 'S'")
                self.faces_cnt = len(faces)

            # 9. 生成的窗口添加说明文字 / Add note on cv2 window
            # self.draw_note(img_rd)
            print('sdad')
            # 10. 按下 'q' 键退出 / Press 'q' to exit
            if kk == ord('q'):
                break
            # self.update_fps()

            show = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
            print(show.data)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

            # self.label.setPixmap(QPixmap.fromImage(showImage))
            self.label.setPixmap(QPixmap("../images/2.png"))

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def getName(self):
        text, okPressed = QInputDialog.getText(self, "Please enter your name", "Your name:", QLineEdit.Normal, "")
        if okPressed and text != '':
            self.newAddName = text
            self.current_face_dir = self.path_photos_from_camera + self.newAddName
            os.makedirs(self.current_face_dir)
            print('\n')
            print("新建的人脸文件夹 / Create folders: ", self.current_face_dir)
            self.ss_cnt = 0  # 将人脸计数器清零 / clear the cnt of faces
            self.press_n_flag = 1  # 已经按下 'n' / have pressed 'n'
            self.capture_button.setVisible(True)
            self.capture_end_button.setVisible(True)
            self.showImageTable()

    def showImageTable(self):
        imagelist = os.listdir("data/data_faces_from_camera/" + self.newAddName)
        for i in range(len(imagelist)):

            newItem = QTableWidgetItem(self.newAddName)
            self.image_tableWidget.setItem(i, 0, newItem)
            newItem = QTableWidgetItem(imagelist[i])
            self.image_tableWidget.setItem(i, 1, newItem)
            newItem = QTableWidgetItem('X')
            self.image_tableWidget.setItem(i, 2, newItem)

    def showImageTable2(self):
        self.image_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 设置图片的大小
        self.image_tableWidget.setIconSize(QSize(300, 200))

        # 设置所有行列宽高数值与图片大小相同
        for i in range(3):  # 让列宽和图片相同
            self.image_tableWidget.setColumnWidth(i, 300)
        for i in range(5):  # 让行高和图片相同
            self.image_tableWidget.setRowHeight(i, 200)

        for k in range(15):
            i = k / 3
            j = k % 3

            # 实例化表格窗口条目
            item = QTableWidgetItem()
            # 用户点击表格时，图片被选中
            item.setFlags(Qt.ItemIsEnabled)
            # 图片路径设置与图片加载
            icon = QIcon(r'.\images\bao%d.png' % k)
            item.setIcon(QIcon(icon))
            # 输出当前进行的条目序号
            print('e/icons/%d.png i=%d  j=%d' % (k, i, j))
            # 将条目加载到相应行列中
            self.image_tableWidget.setItem(i, j, item)



    def capture(self):
        if self.save_flag:
            self.press_s_flag = 1
            self.tipsShow("录入成功！")
        else:
            self.tipsShow("请重试！")
        self.ss_cnt += 1
        self.image_tableWidget.setRowCount(self.image_tableWidget.rowCount() + 1)
        self.showImageTable()


    def capture_end(self):
        self.press_n_flag = 0
        self.tipsHidden()
        self.newAddName = ''
        self.capture_button.setVisible(False)
        self.capture_end_button.setVisible(False)

    def featureExtraction(self):
        # 获取已录入的最后一个人脸序号 / get the num of latest person
        person_list = os.listdir(self.path_photos_from_camera)
        person_cnt = len(person_list)

        with open("data/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            with open("data/person_all.csv", "w", newline="") as csvfile2:
                writer2 = csv.writer(csvfile2)
                for person in range(person_cnt):
                    # Get the mean/average features of face/personX, it will be a list with a length of 128D
                    print(self.path_photos_from_camera + person_list[person])
                    features_mean_personX = return_features_mean_personX(self.path_photos_from_camera + person_list[person])
                    writer.writerow(features_mean_personX)
                    writer2.writerow([person_list[person]])
                    print("特征均值 / The mean of features:", list(features_mean_personX))
                    print('\n')
                print("所有录入人脸数据存入 / Save all the features of faces registered into: data/features_all.csv")
        with open("data/person_all.csv", 'r') as f:
            reader = csv.reader(f)
            self.name_known_list = np.array(list(reader))
        with open("data/features_all.csv", 'r') as f:
            reader = csv.reader(f)
            self.features_known_list = list(reader)
    def faceRecognition(self):
        self.timer_camera.timeout.connect(self.faceRecognitionProcess)

    def faceRecognitionProcess(self):
        flag, img_rd = self.cap.read()        # Get camera video stream
        self.faces = detector(img_rd, 0)         # Use dlib face detector

        self.update_fps()
        self.draw_note(img_rd)


        self.features_camera_list = []
        self.faces_cnt = 0
        self.pos_camera_list = []
        self.name_camera_list = []

        # 5. 检测到人脸 / Face detected
        if len(self.faces) != 0:
            print("检测到人脸")
            # 3. 获取当前捕获到的图像的所有人脸的特征，存储到 self.features_camera_list
            # 3. Get the features captured and save into self.features_camera_list
            for i in range(len(self.faces)):
                shape = predictor(img_rd, self.faces[i])
                self.features_camera_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
            for k in range(len(self.faces)):
                self.name_camera_list.append("unknown")
                self.pos_camera_list.append(tuple([self.faces[k].left(), int(self.faces[k].bottom() + (self.faces[k].bottom() - self.faces[k].top()) / 4)]))
                e_distance_list = []
                for i in range(len(self.features_known_list)):
                    # 如果 person_X 数据不为空
                    if str(self.features_known_list[i][0]) != '0.0':
                        print("with person", self.name_known_list[i], "the e distance: ", end='')
                        # print(self.features_camera_list[k].shape,self.features_known_list[i].shape)
                        e_distance_tmp = return_euclidean_distance(self.features_camera_list[k],
                                                                        self.features_known_list[i])
                        e_distance_list.append(e_distance_tmp)
                    else:
                        # 空数据 person_X
                        e_distance_list.append(999999999)
                # 6. 寻找出最小的欧式距离匹 / F配ind the one with minimum e distance
                similar_person_num = e_distance_list.index(min(e_distance_list))
                print("Minimum e distance with person", self.name_known_list[similar_person_num])

                if min(e_distance_list) < 0.4:
                    self.name_camera_list[k] = self.name_known_list[similar_person_num]
                    print("May be person " + str(self.name_known_list[similar_person_num]))
                else:
                    print("Unknown person")
                # 矩形框 / Draw rectangle
                for kk, d in enumerate(self.faces):
                    # 绘制矩形框
                    cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                  (0, 255, 255), 2)
                print('\n')
            self.faces_cnt = len(self.faces)
            print(self.faces_cnt)
            img_with_name = self.draw_name(img_rd)
            # engine.say(self.name_camera_list[0]+"님, 어서 오세요")
            # engine.runAndWait()
        else:
            print("未检测到")
            img_with_name = img_rd

        show = cv2.cvtColor(img_with_name, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(showImage))
        self.timer_camera.start(10)

    def draw_name(self, img_rd):
        # 在人脸框下面写人脸名字 / Write names under rectangle
        font = ImageFont.truetype("simsun.ttc", 30)
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        print(self.faces_cnt)
        for i in range(self.faces_cnt):
            print(self.pos_camera_list[i],self.name_camera_list[i])
            # cv2.putText(img_rd, self.name_camera_list[i], self.pos_camera_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            # draw.text(xy=self.pos_camera_list[i], text=self.name_camera_list[i],font=font)
            #             # self.tipsShow(self.name_camera_list)
            img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            print("2222")
        print("3333")

        return img_with_name

    def tipsShow(self,message):
        self.tips.setText(message)
        self.tips.setVisible(True)
    def tipsHidden(self):
        self.tips.setText("")
        self.tips.setVisible(False)

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec_())
