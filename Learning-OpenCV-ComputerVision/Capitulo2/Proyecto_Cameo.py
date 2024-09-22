# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:26:02 2021

@author: MBI
"""
import cv2 ,time
import numpy as np 
"""
Nota:
    -Las variables que se declaran con un prefijo delante son tratadas como variables protejidas. Solo puede ser accedidas desde la clase en que son definidas y subclases hijos.
    -Las variables que se declaran con doble prefijo delante son tratadas como variables privadas. Solo  pueden ser accedidas desde la clase en que es definida.
    -Las variables sin prefijos son tratadas como publicas. Son accedidas desde caualquier lugar.
"""
class CaptureManager(object):
    def __init__(self,capture,preWM=None,shouldMP=False):
        self.preWM = preWM
        self.shouldMP = shouldMP
        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None 
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None
    
    @property
    def channel(self):
        return self._channel
    
    @channel.setter
    def channel(self,value):
        if self._channel != value:
            self._channel = value
            self._frame = None
    
    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _,self._frame = self._capture.retrieve(self._frame,self.channel)
        return self._frame 
    
    @property
    def isWritingImage(self):
        return self._imageFilename is not None 
    
    @property 
    def isWritingVideo(self):
        return self._videoFilename is not None 
    
    def enterFrame(self):
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()
    
    def exitFrame(self):
        if self.frame is None:
            self._enteredFrame = False
            return 
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed /timeElapsed
        self._framesElapsed += 1
        if self.preWM is not None:
            if self.shouldMP:
                mirroredFrame = np.fliplr(self._frame)
                self.preWM.show(mirroredFrame)
            else:
                self.preWM.show(self._frame)
        
        if  self.isWritingImage:
            cv2.imwrite(self._imageFilename,self._frame)
            self._imageFilename = None 
        self._writeVideoFrame()
        self._frmae = None
        self._enteredFrame = False
    
    def writeImage(self,filename):
        self._imageFilename = filename 
    
    def startWritingVideo(self,filename,encoding=cv2.VideoWriter_fourcc('M','J','P','G')):
        self._videoFilename = filename
        self._videoEncoding = encoding
    
    def stopWritingVideo(self):
        self._videoFilename = None 
        self._videoEncoding = None 
        self._videoWriter = None 
    
    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return 
        if self._videoWriter is None: 
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0.0:
                if self._framesElapsed < 20:
                    return 
                else:
                    fps = self._fpsEstimate 
            
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename,self._videoEncoding,fps,size)   
            
        self._videoWriter.write(self._frame)

class WindowsManager(object):
    def __init__(self,windowName,keypressCallback=None):
        self.keypressCallback = keypressCallback 
        self._windowName = windowName
        self._isWindowCreated = False
    
    @property 
    def isWindowCreated(self):
        return self._isWindowCreated
    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True 
    
    def show(self,frame):
        cv2.imshow(self._windowName,frame)
        
    def destroyWindows(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False 
    
    def processEvnet(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            self.keypressCallback(keycode)

class Cameo(object):
    def __init__(self):
        self._windowManager = WindowsManager('Cameo',self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0),self._windowManager,True)
    
    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame() 
            frame = self._captureManager.frame 
            if frame is not None:
                pass 
            self._captureManager.exitFrame() 
            self._windowManager.processEvnet() 
    
    def onKeypress(self,keycode):
        if keycode == 32: # Caracter de espacio del tecaldo
            self._captureManager.writeImage("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo2/Screenshot.png")
        elif keycode == 9:# Caracter tab del tecaldo
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo2/Screencast.avi")
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # Caracter escape 
            self._windowManager.destroyWindows() 

if __name__ == '__main__':
    Cameo().run()
            
        

    
    
    
    
    
    
    
    
    
    
    
    
    

