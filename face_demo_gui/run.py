import sys
from PyQt4.QtCore import Qt
from PyQt4.QtCore import SIGNAL
from PyQt4 import QtGui
from PyQt4.QtGui import QLineEdit, QLabel, QFileDialog, QListWidget, QImage
import cPickle
import os
import glob
import cv2
from PyQt4.QtCore import QString
# import enroll_images
# import feature_build
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
import dlib
from imutils import face_utils

def obtain_info():
    if os.path.exists('data/gui_dict.pickle'):
        with open('data/gui_dict.pickle', 'rb') as handle:
            dict_personinfo = cPickle.load(handle)
    else:
        dict_personinfo = dict()

    return dict_personinfo

def save():
    with open('data/gui_dict.pickle', 'wb') as handle:
        cPickle.dump(dict_personinfo, handle)

dict_personinfo = obtain_info()


def opencvDetectFace(img,detector,predictor):
    rectangles = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    rects = detector(gray, 1)
    #rects = detector.detectMultiScale(gray, 1.3, 5)
    for rect in rects:
        landmarks = []
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
        #rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        landmarks.append(rect.left())
        landmarks.append(rect.top())
        landmarks.append(rect.right())
        landmarks.append(rect.bottom())
        landmarks.append(1)
        landmarks.append((int(np.round(np.mean(shape[36:41,:],axis=0))[0])))
        landmarks.append((int(np.round(np.mean(shape[36:41,:],axis=0))[1])))
        landmarks.append((int(np.round(np.mean(shape[42:47,:],axis=0))[0])))
        landmarks.append(int(np.round(np.mean(shape[42:47,:],axis=0)[1])))
        landmarks.append(int(np.round(shape[30])[0]))
        landmarks.append(int(np.round(shape[30])[1]))
        landmarks.append(int(np.round(shape[48])[0]))
        landmarks.append(int(np.round(shape[48])[1]))
        landmarks.append(int(np.round(shape[54])[0]))
        landmarks.append(int(np.round(shape[54])[1]))

        rectangles.append(landmarks)
    return rectangles

def test_create_image(directory, singleimg):
    if singleimg is False:
        direct = directory + "/*"
        images = []
        face_points = 'shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(face_points)
        for image in glob.glob(direct):
            singleimg = []
            frame = cv2.imread(image)
            rectangle = opencvDetectFace(frame, detector, predictor)
            singleimg.append(frame)
            singleimg.append(rectangle)
            images.append(singleimg)
        return images
    else:
        direct = directory
        face_points = 'shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(face_points)
        singleimg = []
        frame = cv2.imread(direct)
        rectangle = opencvDetectFace(frame, detector, predictor)
        singleimg.append(frame)
        singleimg.append(rectangle)
        return singleimg

def refresh_list_elements(ListField, ListSize):
    # dict_personinfo = obtain_info()
    ListField.clear()
    ListSize.setText(str(len(dict_personinfo)) + " Persons Enrolled")

    for key in dict_personinfo.keys():
        ListField.addItem(key)

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50,900, 590)
        self.setWindowTitle("CUBS FaceDemo Enroller")
        self.home()

    def home(self):
        self.imgindex = 0
        self.selectedname = ""
        self.imglen = 0
        self.delbtn = QtGui.QPushButton("Delete Entry", self)
        self.delbtn.resize(100, 30)
        self.delbtn.move(200, 125)
        # self.delbtn.move(0, 0)

        self.rightname = QLabel("", self)
        self.rightname.move(350,5)
        self.rightname.resize(600,20)
        self.rightname.setAlignment(Qt.AlignCenter)

        self.btn = QtGui.QPushButton("Enroll", self)
        # self.btn.clicked.connect()
        self.btn.resize(80,30)
        self.btn.move(10,125)

        self.nextbtn = QtGui.QPushButton("Next", self)
        self.delimagebtn = QtGui.QPushButton("Delete Image", self)
        self.previousbtn = QtGui.QPushButton("Previous", self)
        self.nextbtn.move(750,530)
        self.delimagebtn.move(550,530)
        self.previousbtn.move(450,530)

        self.firstnametext = QLabel(self)
        self.firstnametext.setText("First Name:")
        self.firstnametext.move(10,10)

        self.firstnamefield = QLineEdit(self)
        self.firstnamefield.setObjectName("First Name")
        self.firstnamefield.resize(300, 30)
        self.firstnamefield.move(100,10)

        lastnametext = QLabel(self)
        lastnametext.setText("Last Name:")
        lastnametext.move(10, 50)

        self.lastnamefield = QLineEdit(self)
        self.lastnamefield.setObjectName("First Name")
        self.lastnamefield.resize(300, 30)
        self.lastnamefield.move(100, 50)

        self.dirbtn = QtGui.QPushButton("Choose Directory", self)
        self.dirbtn.resize(140, 30)
        self.dirbtn.move(10, 90)

        self.dirfield = QLineEdit(self)
        self.dirfield.setObjectName("Directory")
        self.dirfield.resize(250,30)
        self.dirfield.move(150, 90)

        self.status = QLabel(self)
        self.status.move(10,563)
        self.status.resize(400,20)
        self.status.setText("Ready")

        self.rebuild = QtGui.QPushButton("Add Image", self)
        self.rebuild.move(650,530)
        self.rebuild.resize(100,30)

        self.listsize = QtGui.QLabel(self)
        self.listsize.setText(" Persons Enrolled")
        self.listsize.resize(200,20)
        self.listsize.move(10,160)

        self.list = QListWidget(self)
        self.list.resize(389,380)
        self.list.move(10,180)
        self.stat("Pulling list")
        refresh_list_elements(self.list, self.listsize)
        self.stat("Ready")

        self.image = QLabel(self)
        self.image.resize(490,490)
        self.image.move(405,30)


        self.list.doubleClicked.connect(self.on_double_click)
        self.connect(self, SIGNAL('triggered()'), self.closeEvent)
        self.connect(self.rebuild, SIGNAL("clicked()"), self.add_image)
        self.connect(self.nextbtn, SIGNAL("clicked()"),self.next_image)
        self.connect(self.previousbtn, SIGNAL("clicked()"), self.previous_image)
        self.connect(self.delimagebtn, SIGNAL("clicked()"), self.delete_image)
        self.connect(self.dirbtn, SIGNAL("clicked()"), self.opendir)
        self.connect(self.delbtn, SIGNAL("clicked()"), self.button_delete)
        self.connect(self.btn, SIGNAL("clicked()"), self.button_click)
        self.show()

    def closeEvent(self, event):
        self.stat("Saving Changes")
        save()
        print "Changes saved"
        print "Rebuilding Vectors"
        self.rebuild_vector()
        self.destroy()

    def rebuild_vector(self):
        # feature_build.generate_vectors()
        pass

    def stat(self, stat):
        self.status.setText(stat)

    def add_image(self):
        self.dialog = QFileDialog(self)
        directory = self.dialog.getOpenFileName(self, 'Select Directory')
        if directory:
            self.dirfield.setText(directory)
        img_directory = str(self.dirfield.text())
        picarray = test_create_image(img_directory, True)
        # enroll_images.images_to_array(img_directory, True)
        dict_personinfo[self.selectedname].append(picarray)
        refresh_list_elements(self.list, self.listsize)
        self.init_images()
        self.dirfield.setText('')

    def next_image(self):
        check = self.imgindex + 1
        if(check >= self.imglen):
            self.display_image(0)
            self.imgindex = 0
        else:
            self.display_image(check)
            self.imgindex += 1

    def previous_image(self):
        check = self.imgindex - 1
        if (check < 0):
            self.display_image(self.imglen-1)
            self.imgindex = self.imglen-1
        else:
            self.display_image(check)
            self.imgindex -= 1

    def delete_image(self):
        # dict_personinfo = obtain_info()
        dict_personinfo[self.selectedname].pop(self.imgindex)
        if(len(dict_personinfo[self.selectedname])==0):
            dict_personinfo.pop(self.selectedname, None)
            # with open('data/gui_dict.pickle', 'wb') as handle:
            #     cPickle.dump(dict_personinfo, handle)
            self.stat("Deleted " + self.selectedname)
            refresh_list_elements(self.list, self.listsize)
            self.image.clear()
            self.rightname.setText("")
        else:
            # with open('data/gui_dict.pickle', 'wb') as handle:
            #     cPickle.dump(dict_personinfo, handle)
            self.stat("Deleted image for " + self.selectedname)
            refresh_list_elements(self.list, self.listsize)
            self.init_images()


    def display_image(self, index=0):
        img = self.get_images()
        self.imglen = len(img)
        img = img[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qimg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qimg)
        pixmap = pixmap.scaled(self.image.size())
        self.image.setPixmap(pixmap)

    def get_images(self):
        # dict_personinfo = obtain_info()
        arr_images = []
        for i in range(len(dict_personinfo[self.selectedname])):
            frame = dict_personinfo[self.selectedname][i][0]
            rectangle = dict_personinfo[self.selectedname][i][1]
            img = np.array(Image.fromarray(frame))
            cv2.rectangle(img, (int(rectangle[0][0]), int(rectangle[0][3])), (int(rectangle[0][2]),int(rectangle[0][1])), (0,255,0), 40)
            cv2.circle(img, (int(rectangle[0][5]), int(rectangle[0][6])), 10, (0,255,0), 40)
            cv2.circle(img, (int(rectangle[0][7]), int(rectangle[0][8])), 10, (0, 255, 0), 40)
            cv2.circle(img, (int(rectangle[0][9]), int(rectangle[0][10])), 10, (0, 255, 0), 40)
            cv2.circle(img, (int(rectangle[0][11]), int(rectangle[0][12])), 10, (0, 255, 0), 40)
            cv2.circle(img, (int(rectangle[0][13]), int(rectangle[0][14])), 10, (0, 255, 0), 40)

            print(rectangle)
            arr_images.append(img)
        self.imglen = len(arr_images)
        return arr_images

    def on_double_click(self):
        self.selectedname = str(self.list.currentItem().text())
        self.rightname.setText(self.selectedname)
        self.init_images()


    def init_images(self):
        self.imgindex = 0
        img = self.get_images()
        img = img[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qimg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qimg)
        pixmap = pixmap.scaled(self.image.size())
        self.image.setPixmap(pixmap)

    def opendir(self):
        self.dialog = QFileDialog(self)
        directory = self.dialog.getExistingDirectory(self, 'Select Directory')
        if directory:
            self.dirfield.setText(directory)

    def button_delete(self):
        if (str(self.list.currentItem().text())!= None):
            # dict_personinfo = obtain_info()
            dict_personinfo.pop(str(self.list.currentItem().text()), None)
            # with open('data/gui_dict.pickle', 'wb') as handle:
            #     cPickle.dump(dict_personinfo, handle)
            self.stat("Deleted " + str(self.list.currentItem().text()))
            refresh_list_elements(self.list, self.listsize)
            self.image.clear()
            self.rightname.clear()
        else:
            pass

    def button_click(self):
        self.stat("Pulling list")
        # dict_personinfo = obtain_info()
        self.stat("Ready")

        name = ['','']
        name[0] = str(self.firstnamefield.text())
        name[1] = str(self.lastnamefield.text())
        name[0] = name[0].replace(" ", "")
        name[1] = name[1].replace(" ", "")
        img_directory = str(self.dirfield.text())

        self.stat("Detecting Face")
        if img_directory == "" or len(name[0]) == 0:
            picarray = []
        else:
            picarray = test_create_image(img_directory, False) # Just a testing filler for now
            # picarray = enroll_images.images_to_array(img_directory, False)

        # feature_build.generate_vectors()

        if len(picarray) != 0:
            dict_personinfo[name[0] + " " + name[1]] = {}
            dict_personinfo[name[0] + " " + name[1]] = picarray

            if img_directory == "" or len(name[0])==0:
                self.stat("Failed to add " + name[0] + " " + name[1])
            else:
                self.stat("Writing")
                # with open('data/gui_dict.pickle', 'wb') as handle:
                #     cPickle.dump(dict_personinfo, handle)
                self.stat("Added " + name[0] + " " + name[1] + " with " + str(len(picarray)) + " images")
        else:
            self.stat("Failed to add " + name[0] + " " + name[1])
        # feature_build.generate_vectors()
        self.firstnamefield.setText('')
        self.lastnamefield.setText('')
        self.dirfield.setText('')
        # self.list.addItem(name[0] + " " + name[1])
        refresh_list_elements(self.list, self.listsize)


def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())


run()

