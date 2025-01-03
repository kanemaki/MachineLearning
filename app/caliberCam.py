# Documentação - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import numpy as np
import cv2

cap = cv2.VideoCapture('../imagens/xadrez.mp4')
# criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# praparar pontos dos objetos, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:7].T.reshape(-1,2)

# Matrizes para armazenar pontos de objetos e pontos de imagem de todas as imagens.
objpoints = [] # Ponto 3D no espaço do mundo real.
imgpoints = [] # Pontos 2d no plano da imagem.

while True:
    imagem = cv2.imread('../imagens/xadrez.jpg')
    _,frame = cap.read()
    gray = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6,7),None)

    # Se encontrado, adicione pontos de objeto, pontos de imagem (após refiná-los)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Desenhe e exiba os cantos
        img = cv2.drawChessboardCorners(imagem, (6,7), corners2,ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        h,  w = imagem.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

            print("total error: ", mean_error/len(objpoints))
        # cortar a imagem
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('calibresult',dst)

    #cv2.imshow('imagem',imagem)
    cv2.imshow('frame',frame)
    k=cv2.waitKey(1)
    if k == ord('q'):
       exit()

cv2.destroyAllWindows()
