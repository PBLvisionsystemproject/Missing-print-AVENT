#from os import wait4
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QPushButton
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
import cv2
import numpy as np
from matplotlib import pyplot as plt

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Template Matching'
        self.left = 50
        self.top = 50
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create widget

        button = QPushButton('Select Image', self)
        button.setToolTip('This is load picture button')
        button.move(10, 10)
        button.clicked.connect(self.on_click)

        self.TestImage = QLabel(self)
        self.TestImage.move(10,50)

        self.TestOutput = QLabel(self)
        self.TestOutput.move(110,15)

        self.Test40 = QLabel(self)
        self.Test40.move(150,15)

        self.Test60 = QLabel(self)
        self.Test60.move(110,15)
        


        #self.resize(pixmap.width(), pixmap.height())

        self.show()

    @pyqtSlot()
    def on_click(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', 'D:\PBL\PBL AVENT logo\OKE PATOKAN', "Image file(*.jpg)")
        imagePath = image[0]
        

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        i = 0
        j = 0
        k = 0
        l = 0

        img_rgb = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template1 = cv2.imread('OKE_PATOKAN_AV.jpg',0) 
        template2 = cv2.imread('OKE_PATOKAN_E.jpg' ,0)
        template3 = cv2.imread('OKE_PATOKAN_N.jpg' ,0)
        template4 = cv2.imread('OKE_PATOKAN_T.jpg' ,0)
        template5 = cv2.imread('OKE_PATOKAN_P.jpg' ,0)
        template6 = cv2.imread('OKE_PATOKAN_H.jpg' ,0)
        template7 = cv2.imread('OKE_PATOKAN_I.jpg' ,0)
        template8 = cv2.imread('OKE_PATOKAN_L.jpg' ,0)
        template9 = cv2.imread('OKE_PATOKAN_S.jpg' ,0)
        template10 = cv2.imread('p_good2.jpg' ,0)
        template11 = cv2.imread('h_good2.jpg' ,0)

        #WEIGHT HEIGHT
        w1, h1 = template1.shape[::-1]
        w2, h2 = template2.shape[::-1]
        w3, h3 = template3.shape[::-1]
        w4, h4 = template4.shape[::-1]
        w5, h5 = template5.shape[::-1]
        w6, h6 = template6.shape[::-1]
        w7, h7 = template7.shape[::-1]
        w8, h8 = template8.shape[::-1]
        w9, h9 = template9.shape[::-1]
        w10, h10 = template10.shape[::-1]
        w11, h11 = template11.shape[::-1]

        #RES
        res1 = cv2.matchTemplate(img_gray,template1,cv2.TM_CCOEFF_NORMED)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        res2 = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        res3 = cv2.matchTemplate(img_gray,template3,cv2.TM_CCOEFF_NORMED)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
        res4 = cv2.matchTemplate(img_gray,template4,cv2.TM_CCOEFF_NORMED)
        min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)
        res5 = cv2.matchTemplate(img_gray,template5,cv2.TM_CCOEFF_NORMED)
        min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)
        res6 = cv2.matchTemplate(img_gray,template6,cv2.TM_CCOEFF_NORMED)
        min_val6, max_val6, min_loc6, max_loc6 = cv2.minMaxLoc(res6)
        res7 = cv2.matchTemplate(img_gray,template7,cv2.TM_CCOEFF_NORMED)
        min_val7, max_val7, min_loc7, max_loc7 = cv2.minMaxLoc(res7)
        res8 = cv2.matchTemplate(img_gray,template8,cv2.TM_CCOEFF_NORMED)
        min_val8, max_val8, min_loc8, max_loc8 = cv2.minMaxLoc(res8)
        res9 = cv2.matchTemplate(img_gray,template9,cv2.TM_CCOEFF_NORMED)
        min_val9, max_val9, min_loc9, max_loc9 = cv2.minMaxLoc(res9)
        res10 = cv2.matchTemplate(img_gray,template10,cv2.TM_CCOEFF_NORMED)
        min_val10, max_val10, min_loc10, max_loc10 = cv2.minMaxLoc(res10)
        res11 = cv2.matchTemplate(img_gray,template11,cv2.TM_CCOEFF_NORMED)
        min_val11, max_val11, min_loc11, max_loc11 = cv2.minMaxLoc(res11)

        #THRESHOLD 
        threshold1 = 0.95     # AV
        threshold2 = 0.96     # E
        threshold3 = 0.95     # N
        threshold4 = 0.926    # T
        threshold5 = 0.89     # P
        threshold6 = 0.912    # H
        threshold7 = 0.95     # I
        threshold8 = 0.969    # L
        threshold9 = 0.93     # S
        threshold10 = 0.8     # p
        threshold11 = 0.8     # h

        #RECTANGLE
        loc1 = np.where( res1 >= threshold1)
        for pt in zip(*loc1[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w1, pt[1] + h1), (0,0,255), 2)
            max_val1 = round(max_val1, ndigits=2)
            txt1 = str(max_val1)
            cv2.putText(img_rgb, txt1, (max_loc1[0], max_loc1[1] - 10) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            a = 1

        loc2 = np.where( res2 >= threshold2)
        for pt in zip(*loc2[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w2, pt[1] + h2), (0,0,255), 2)
            max_val2 = round(max_val2, ndigits=2)
            txt2 = str(max_val2)
            cv2.putText(img_rgb, txt2, (max_loc2[0], max_loc2[1] - 10) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            b = 1

        loc3 = np.where( res3 >= threshold3)
        for pt in zip(*loc3[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w3, pt[1] + h3), (0,0,255), 2)
            max_val3 = round(max_val3, ndigits=2)
            txt3 = str(max_val3)
            cv2.putText(img_rgb, txt3, (max_loc3[0], max_loc3[1] - 10) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            c = 1

        loc4 = np.where( res4 >= threshold4)
        for pt in zip(*loc4[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w4, pt[1] + h4), (0,0,255), 2)
            max_val4 = round(max_val4, ndigits=2)
            txt4 = str(max_val4)
            cv2.putText(img_rgb, txt4, (max_loc4[0], max_loc4[1] - 10) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            d = 1

        loc5 = np.where( res5 >= threshold5)
        for pt in zip(*loc5[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w5, pt[1] + h5), (0,0,255), 2)
            max_val5 = round(max_val5, ndigits=2)
            txt5 = str(max_val5)
            cv2.putText(img_rgb, txt5, (max_loc5[0] + 20, max_loc5[1] - 10) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            e = 1

        loc6 = np.where( res6 >= threshold6)
        for pt in zip(*loc6[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w6, pt[1] + h6), (0,0,255), 2)
            max_val6 = round(max_val6, ndigits=2)
            txt6 = str(max_val6)
            cv2.putText(img_rgb, txt6, (max_loc6[0]+ 10, max_loc6[1] - 15) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            f = 1

        loc7 = np.where( res7 >= threshold7)
        for pt in zip(*loc7[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w7, pt[1] + h7), (0,0,255), 2)
            max_val7 = round(max_val7, ndigits=2)
            txt7 = str(max_val7)
            cv2.putText(img_rgb, txt7, (max_loc7[0], max_loc7[1] - 10) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            g = 1

        loc8 = np.where( res8 >= threshold8)
        for pt in zip(*loc8[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w8, pt[1] + h8), (0,0,255), 2)
            max_val8 = round(max_val8, ndigits=2)
            txt8 = str(max_val8)
            cv2.putText(img_rgb, txt8, (max_loc8[0], max_loc8[1] - 30) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            i = 1
            
        loc9 = np.where( res9 >= threshold9)
        for pt in zip(*loc9[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w9, pt[1] + h9), (0,0,255), 2)
            max_val9 = round(max_val9, ndigits=2)
            txt9 = str(max_val9)
            cv2.putText(img_rgb, txt9, (max_loc9[0], max_loc9[1] - 10) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            j = 1

        loc10 = np.where( res10 >= threshold10)
        for pt in zip(*loc10[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w10, pt[1] + h10), (0,0,255), 2)
            max_val10 = round(max_val10, ndigits=2)
            txt10 = str(max_val10)
            cv2.putText(img_rgb, txt10, (max_loc10[0], max_loc10[1] - 10) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            k = 1

        loc11 = np.where( res11 >= threshold11)
        for pt in zip(*loc11[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w11, pt[1] + h11), (0,0,255), 2)
            max_val11 = round(max_val11, ndigits=2)
            txt11 = str(max_val11)
            cv2.putText(img_rgb, txt11, (max_loc11[0], max_loc11[1] - 10) , cv2.FONT_ITALIC, 0.6, (0,0,255), 2)
            l = 1



        cv2.imwrite('res.jpg',img_rgb)
        print('', a, b, c, d, e, f, g, i, j)

        if a > 0 and b > 0 and c > 0 and d > 0 and e > 0 and f > 0 and g > 0 and i > 0 and j > 0 and k > 0 and l > 0 :
            self.TestOutput.setText('BARANG GOOD')
            self.TestOutput.adjustSize() 
        else :
            self.TestOutput.setText('BARANG NOT GOOD')
            self.TestOutput.adjustSize()
        
        #MENAMPILIN PERSEN
        print('', max_val1)
        print('', max_val2)
        print('', max_val3)
        print('', max_val4)
        print('', max_val5)
        print('', max_val6)
        print('', max_val7)
        print('', max_val8)
        print('', max_val9)
        print('', max_loc10)
        print('', max_loc11)

        pixmap = QPixmap('res.jpg')
        self.TestImage.setPixmap(pixmap)
        
        self.TestImage.adjustSize()                       # <---
        self.resize(pixmap.size())
        self.adjustSize()

        cv2.waitKey(0)
        cv2.destroyAllWindows() 
 


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())