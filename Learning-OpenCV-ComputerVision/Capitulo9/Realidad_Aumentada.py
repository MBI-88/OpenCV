# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:13:50 2021

@author: MBI
"""
import cv2
import numpy as np
import timeit 
import math
#%%
# Entendiendo el metodo cv2.solvePnPRansac
"""
Este metodo implementa una solucion llamada Perspectiva de m Puntos (PnP).
Dado un set de n ejecutores unicos entre 3D y 2D puntos, solo con los parametros de la lente y la camara que genero esta proyeccion  2D de un puntos 3D, la solucion intenta estimar la posicion 6DOF de el objeto 3D realtivo a la camara.

Principio de funcionamiento de cv2.solvePnPRansac:
    Implenta el algoritmo Ransac el cual es un acercamiento de iteracion de propositos generales diseñado para confrontar con un set de entradas que contiene valores extremos que en nuestro  caso son malos ejecutores. Cada iteracion Ransac encuentra una solucion potencial que minimize el error medio de la entrada. Entonces antes de la siguiente iteracion cualquier resultado con un error grande es marcado como valor extremo y es eliminado, este proceso continua hasta que se converja y no se encuentren mas valores extremos.

Parametros de el metodo:
    
    .retval : Este es booleno. Si es verdadero la convergencia se logro.
    .rvec : Este contiene el array de rx,ry y rz , los tres grados rotacionales de libertad en el 6DOF.
    .tvec : Este contiene el array de tx,ty y tz , los tres grados de libertad transicional en el 6DOF.
    .inliers : Contiene los indices de los puntos de entrada si se convergio (in objectPoints y imagePoints).
    .objectPoints : Array de puntos de 3D que representa los puntos claves del objeto cundo no hay traslacion ni rotacion.
    .imagePoints : Array de puntos de 2D que representa los puntos claves de los ejecutores en la imagen.
    .cameraMatrix : Matriz  de la camara en 2D.
    .distCoeffs : Array de coeficiente de distorcion.
    .useExtrinsicGuess : Es booleano. Si es verdadero la solucion de valores en el rvec y tvec son inicialmente estimados y entonces intenta encontrar una solucion que es cercana  a esa.
    .iterationsCount : Es el maximo numero de iteraciones que el soucionador aceptara, si un punto tiene un error  de projeccion mayor que este , el solucionador  lo trata como valor extremo.
    .reprojectionError : Maximo valor de  errror que el solucionador aceptara.
    .confidence : El solucionador intenta converger sobre una solucion que tiene una puntuacion de confidencialidad mayor o igual a este valor.
    .inliers : Si el solucionador converge , pondra los indices de los puntos inliers en este array.
    .flags : La bandera expecifica el algoritmo del solucionador. Por defecto es cv2.SOLVEPNP_ITERATIVE que minimiza la reprojeccion del error y no tiene restricciones especiales, es generalmente la mejor opcion. Como una alternativa esta cv2.SOLVEPNP_IPPE pero es restringida a objetos planos.

"""
#%%
# Implementacion

def convert_to_gray(src,dst=None):
    weight = 1.0/3.0
    m = np.array([[weight,weight,weight]],np.float32)
    return cv2.transform(src,m,dst)

def map_point_onto_plane(point_2D,image_size,image_scale):
    x,y = point_2D
    w,h = image_size
    return (image_scale * (x - 0.5 * w),image_scale * (y - 0.5 * h),0.0)

def map_points_to_plane(points_2D,image_size,image_real_height):
    w,h = image_size
    image_scale = image_real_height/h
    points_3D = [map_point_onto_plane(point_2D, image_size, image_scale) for point_2D in points_2D]
    return np.array(points_3D,np.float32)

def map_vertices_to_plane(image_size,image_real_height):
    w,h = image_size
    vertices_2D = [(0,0),(w,0),(w,h),(0,h)]
    vertex_indices_by_face = [[0,1,2,3]]
    vertices_3D = map_points_to_plane(vertices_2D, image_size, image_real_height)
    return vertices_3D,vertex_indices_by_face


class ImageTrackingDemo():


    def __init__(self, capture, diagonal_fov_degrees=70.0,
                 target_fps=25.0,
                 reference_image_path='reference_image.png',
                 reference_image_real_height=1.0):

        self._capture = capture
        success, trial_image = capture.read()
        if success:
            # Use the actual image dimensions.
            h, w = trial_image.shape[:2]
        else:
            # Use the nominal image dimensions.
            w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._image_size = (w, h)

        diagonal_image_size = (w ** 2.0 + h ** 2.0) ** 0.5
        diagonal_fov_radians = \
            diagonal_fov_degrees * math.pi / 180.0
        focal_length = 0.5 * diagonal_image_size / math.tan(
            0.5 * diagonal_fov_radians)
        self._camera_matrix = np.array(
            [[focal_length, 0.0, 0.5 * w],
             [0.0, focal_length, 0.5 * h],
             [0.0, 0.0, 1.0]], np.float32)

        self._distortion_coefficients = None

        self._rotation_vector = None
        self._translation_vector = None

        self._kalman = cv2.KalmanFilter(18, 6)

        self._kalman.processNoiseCov = np.identity(
            18, np.float32) * 1e-5
        self._kalman.measurementNoiseCov = np.identity(
            6, np.float32) * 1e-2
        self._kalman.errorCovPost = np.identity(
            18, np.float32)

        self._kalman.measurementMatrix = np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            np.float32)
        # Se pon 1.0 en el indice de acada variabl (Tx,Ty,Tz,Rx,Ry,Rz)
        self._init_kalman_transition_matrix(target_fps)

        self._was_tracking = False

        self._reference_image_real_height = \
            reference_image_real_height
        reference_axis_length = 0.5 * reference_image_real_height

        #-----------------------------------------------------------------------------
        # BEWARE!
        #-----------------------------------------------------------------------------
        #
        # OpenCV's coordinate system has non-standard axis directions:
        #   +X:  object's left; viewer's right from frontal view
        #   +Y:  down
        #   +Z:  object's backward; viewer's forward from frontal view
        #
        # Negate them all to convert to right-handed coordinate system (like OpenGL):
        #   +X:  object's right; viewer's left from frontal view
        #   +Y:  up
        #   +Z:  object's forward; viewer's backward from frontal view
        #
        #-----------------------------------------------------------------------------
        self._reference_axis_points_3D = np.array(
            [[0.0, 0.0, 0.0],
             [-reference_axis_length, 0.0, 0.0],
             [0.0, -reference_axis_length, 0.0],
             [0.0, 0.0, -reference_axis_length]], np.float32)

        self._bgr_image = None
        self._gray_image = None
        self._mask = None
       

        # Create and configure the feature detector.
        patchSize = 31
        self._feature_detector = cv2.ORB_create(
            nfeatures=250, scaleFactor=1.2, nlevels=16,
            edgeThreshold=patchSize, patchSize=patchSize)

        bgr_reference_image = cv2.imread(
            reference_image_path, cv2.IMREAD_COLOR)
        reference_image_h, reference_image_w = \
            bgr_reference_image.shape[:2]
        reference_image_resize_factor = \
            (2.0 * h) / reference_image_h
        bgr_reference_image = cv2.resize(
            bgr_reference_image, (0, 0), None,
            reference_image_resize_factor,
            reference_image_resize_factor, cv2.INTER_CUBIC)
        gray_reference_image = convert_to_gray(bgr_reference_image)
        reference_mask = np.empty_like(gray_reference_image)

        # Find keypoints and descriptors for multiple segments of
        # the reference image.
        reference_keypoints = []
        self._reference_descriptors = np.empty(
            (0, 32), np.uint8)
        num_segments_y = 6
        num_segments_x = 6
        for segment_y, segment_x in np.ndindex(
                (num_segments_y, num_segments_x)):
            y0 = reference_image_h * \
                segment_y // num_segments_y - patchSize
            x0 = reference_image_w * \
                segment_x // num_segments_x - patchSize
            y1 = reference_image_h * \
                (segment_y + 1) // num_segments_y + patchSize
            x1 = reference_image_w * \
                (segment_x + 1) // num_segments_x + patchSize
            reference_mask.fill(0)
            cv2.rectangle(
                reference_mask, (x0, y0), (x1, y1), 255, cv2.FILLED)
            more_reference_keypoints, more_reference_descriptors = \
                self._feature_detector.detectAndCompute(
                    gray_reference_image, reference_mask)
            if more_reference_descriptors is None:
                # No keypoints were found for this segment.
                continue
            reference_keypoints += more_reference_keypoints
            self._reference_descriptors = np.vstack(
                (self._reference_descriptors,
                 more_reference_descriptors))

        cv2.drawKeypoints(
            gray_reference_image, reference_keypoints,
            bgr_reference_image,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        ext_i = reference_image_path.rfind('.')
        reference_image_keypoints_path = \
            reference_image_path[:ext_i] + '_keypoints' + \
            reference_image_path[ext_i:]
        cv2.imwrite(
            reference_image_keypoints_path, bgr_reference_image)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6, key_size=12,
                            multi_probe_level=1)
        search_params = dict()
        self._descriptor_matcher = cv2.FlannBasedMatcher(
            index_params, search_params)
        self._descriptor_matcher.add([self._reference_descriptors])

        reference_points_2D = [keypoint.pt
                               for keypoint in reference_keypoints]
        self._reference_points_3D = map_points_to_plane(
            reference_points_2D, gray_reference_image.shape[::-1],
            reference_image_real_height)

        (self._reference_vertices_3D,
         self._reference_vertex_indices_by_face) = \
            map_vertices_to_plane(
                    gray_reference_image.shape[::-1],
                    reference_image_real_height)


    def run(self):

        num_images_captured = 0
        start_time = timeit.default_timer()

        while cv2.waitKey(1) != 27:  # Escape
            success, self._bgr_image = self._capture.read(
                self._bgr_image)
            if success:
                num_images_captured += 1
                self._track_object()
                cv2.imshow('Image Tracking', self._bgr_image)
            delta_time = timeit.default_timer() - start_time
            if delta_time > 0.0:
                fps = num_images_captured / delta_time
                self._init_kalman_transition_matrix(fps)
    
    def _track_object(self):

        self._gray_image = convert_to_gray(
            self._bgr_image, self._gray_image)

        if self._mask is None:
            self._mask = np.full_like(self._gray_image, 255)

        keypoints, descriptors = \
            self._feature_detector.detectAndCompute(
                self._gray_image, self._mask)

        # Find the 2 best matches for each descriptor.
        matches = self._descriptor_matcher.knnMatch(descriptors, 2)

        # Filter the matches based on the distance ratio test.
        good_matches = [
            match[0] for match in matches
            if len(match) > 1 and \
        		match[0].distance < 0.6 * match[1].distance
        ]

        # Select the good keypoints and draw them in red.
        good_keypoints = [keypoints[match.queryIdx]
                          for match in good_matches]
        cv2.drawKeypoints(self._gray_image, good_keypoints,
                          self._bgr_image, (0, 0, 255))

        min_good_matches_to_start_tracking = 8
        min_good_matches_to_continue_tracking = 6
        num_good_matches = len(good_matches)

        if num_good_matches < min_good_matches_to_continue_tracking:
            self._was_tracking = False
            self._mask.fill(255)

        elif num_good_matches >= \
                min_good_matches_to_start_tracking or \
                    self._was_tracking:

            # Select the 2D coordinates of the good matches.
            # They must be in an array of shape (N, 1, 2).
            good_points_2D = np.array(
                [[keypoint.pt] for keypoint in good_keypoints],
                np.float32)

            # Select the 3D coordinates of the good matches.
            # They must be in an array of shape (N, 1, 3).
            good_points_3D = np.array(
                [[self._reference_points_3D[match.trainIdx]]
                 for match in good_matches],
                np.float32)

            # Solve for the pose and find the inlier indices.
            (success, self._rotation_vector,
             self._translation_vector, inlier_indices) = \
                cv2.solvePnPRansac(good_points_3D, good_points_2D,
                                   self._camera_matrix,
                                   self._distortion_coefficients,
                                   self._rotation_vector,
                                   self._translation_vector,
                                   useExtrinsicGuess=False,
                                   iterationsCount=100,
                                   reprojectionError=8.0,
                                   confidence=0.99,
                                   flags=cv2.SOLVEPNP_ITERATIVE)

            if success:

                if not self._was_tracking:
                    self._init_kalman_state_matrices()
                self._was_tracking = True

                self._apply_kalman()

                # Select the inlier keypoints.
                inlier_keypoints = [good_keypoints[i]
                                    for i in inlier_indices.flat]

                # Draw the inlier keypoints in green.
                cv2.drawKeypoints(self._bgr_image, inlier_keypoints,
                                  self._bgr_image, (0, 255, 0))

                # Draw the axes of the tracked object.
                self._draw_object_axes()

                # Make and draw a mask around the tracked object.
                self._make_and_draw_object_mask()
    
    def _init_kalman_transition_matrix(self, fps):

        if fps <= 0.0:
            return

        # Velocity transition rate
        vel = 1.0 / fps

        # Acceleration transition rate
        acc = 0.5 * (vel ** 2.0)

        self._kalman.transitionMatrix = np.array(
            [[1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            np.float32)
    
    def _init_kalman_state_matrices(self):

        t_x, t_y, t_z = self._translation_vector.flat
        r_x, r_y, r_z = self._rotation_vector.flat

        self._kalman.statePre = np.array(
            [[t_x], [t_y], [t_z],
             [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0],
             [r_x], [r_y], [r_z],
             [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0]], np.float32)
        self._kalman.statePost = np.array(
            [[t_x], [t_y], [t_z],
             [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0],
             [r_x], [r_y], [r_z],
             [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0]], np.float32)
    
    def _apply_kalman(self):

        self._kalman.predict()

        t_x, t_y, t_z = self._translation_vector.flat
        r_x, r_y, r_z = self._rotation_vector.flat

        estimate = self._kalman.correct(np.array(
            [[t_x], [t_y], [t_z],
             [r_x], [r_y], [r_z]], np.float32))

        self._translation_vector = estimate[0:3]
        self._rotation_vector = estimate[9:12]
    
    def _draw_object_axes(self):

        points_2D, jacobian = cv2.projectPoints(
            self._reference_axis_points_3D, self._rotation_vector,
            self._translation_vector, self._camera_matrix,
            self._distortion_coefficients)

        origin = (int(points_2D[0, 0, 0]), int(points_2D[0, 0, 1]))
        right = (int(points_2D[1, 0, 0]), int(points_2D[1, 0, 1]))
        up = (int(points_2D[2, 0, 0]), int(points_2D[2, 0, 1]))
        forward = (int(points_2D[3, 0, 0]), int(points_2D[3, 0, 1]))

        # Draw the X axis in red.
        cv2.arrowedLine(self._bgr_image, origin, right, (0, 0, 255))

        # Draw the Y axis in green.
        cv2.arrowedLine(self._bgr_image, origin, up, (0, 255, 0))

        # Draw the Z axis in blue.
        cv2.arrowedLine(
            self._bgr_image, origin, forward, (255, 0, 0))
    
    def _make_and_draw_object_mask(self):

        # Project the object's vertices into the scene.
        vertices_2D, jacobian = cv2.projectPoints(
            self._reference_vertices_3D, self._rotation_vector,
            self._translation_vector, self._camera_matrix,
            self._distortion_coefficients)
        vertices_2D = vertices_2D.astype(np.int32)

        # Make a mask based on the projected vertices.
        self._mask.fill(0)
        for vertex_indices in \
                self._reference_vertex_indices_by_face:
            cv2.fillConvexPoly(
                self._mask, vertices_2D[vertex_indices], 255)

        # Draw the mask in semi-transparent yellow.
        cv2.subtract(
            self._bgr_image, 48, self._bgr_image, self._mask)
    
        


def main():

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    diagonal_fov_degrees = 70.0
    target_fps = 25.0

    demo = ImageTrackingDemo(
        capture, diagonal_fov_degrees, target_fps,reference_image_path="C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo9/reference_image.png")
    demo.run()


if __name__ == '__main__':
    main()  
    cv2.destroyAllWindows()
#%%