# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1094, 792)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../images/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolTip("")
        MainWindow.setWhatsThis("")
        MainWindow.setIconSize(QtCore.QSize(50, 50))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.photoAdd_Button = QtWidgets.QPushButton(self.centralwidget)
        self.photoAdd_Button.setGeometry(QtCore.QRect(830, 70, 161, 41))
        self.photoAdd_Button.setObjectName("photoAdd_Button")
        self.newAdd_Button = QtWidgets.QPushButton(self.centralwidget)
        self.newAdd_Button.setGeometry(QtCore.QRect(630, 70, 161, 41))
        self.newAdd_Button.setObjectName("newAdd_Button")
        self.tips = QtWidgets.QLabel(self.centralwidget)
        self.tips.setGeometry(QtCore.QRect(40, 590, 351, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.tips.setFont(font)
        self.tips.setText("")
        self.tips.setObjectName("tips")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 148, 601, 471))
        self.label.setObjectName("label")
        self.capture_button = QtWidgets.QPushButton(self.centralwidget)
        self.capture_button.setGeometry(QtCore.QRect(240, 630, 75, 23))
        self.capture_button.setObjectName("capture_button")
        self.capture_end_button = QtWidgets.QPushButton(self.centralwidget)
        self.capture_end_button.setGeometry(QtCore.QRect(330, 630, 75, 23))
        self.capture_end_button.setObjectName("capture_end_button")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(610, 120, 481, 281))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.image_tableWidget = QtWidgets.QTableWidget(self.verticalLayoutWidget)
        self.image_tableWidget.setStyleSheet("ds")
        self.image_tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.image_tableWidget.setObjectName("image_tableWidget")
        self.image_tableWidget.setColumnCount(3)
        self.image_tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.image_tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.image_tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.image_tableWidget.setHorizontalHeaderItem(2, item)
        self.verticalLayout.addWidget(self.image_tableWidget)
        self.feature_extraction_Button = QtWidgets.QPushButton(self.centralwidget)
        self.feature_extraction_Button.setGeometry(QtCore.QRect(630, 430, 161, 41))
        self.feature_extraction_Button.setObjectName("feature_extraction_Button")
        self.face_recognition_Button = QtWidgets.QPushButton(self.centralwidget)
        self.face_recognition_Button.setGeometry(QtCore.QRect(630, 500, 161, 41))
        self.face_recognition_Button.setObjectName("face_recognition_Button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1094, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "人脸检测系统"))
        MainWindow.setStatusTip(_translate("MainWindow", "任务栏提示信息"))
        self.photoAdd_Button.setText(_translate("MainWindow", "사진 추가"))
        self.newAdd_Button.setText(_translate("MainWindow", "신규 가입"))
        self.label.setText(_translate("MainWindow", "video"))
        self.capture_button.setText(_translate("MainWindow", "캡처"))
        self.capture_end_button.setText(_translate("MainWindow", "완료"))
        item = self.image_tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "name"))
        item = self.image_tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "image"))
        item = self.image_tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "operation"))
        self.feature_extraction_Button.setText(_translate("MainWindow", "특징 추출"))
        self.face_recognition_Button.setText(_translate("MainWindow", "안면 인식"))
