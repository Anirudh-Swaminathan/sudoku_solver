import cv2
import numpy as np
import operator

def findCorners(img):

	_, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	polygon = contours[0]

	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

def displayRects(inImg, rects, colour=255):
	img = inImg.copy()
	for rect in rects:
		img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
	showImage(img)
	return img

def distance(p1, p2):
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))

def crop(img, crop_rect):

	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	side = max([
		distance(bottom_right, top_right),
		distance(top_left, bottom_left),
		distance(bottom_right, bottom_left),
		distance(top_left, top_right)
	])

	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
	m = cv2.getPerspectiveTransform(src, dst)

	return cv2.warpPerspective(img, m, (int(side), int(side)))

def gridDetect(img):

	squares = []
	side = img.shape[:1]
	side = side[0] / 9
	for i in range(9):
		for j in range(9):
			p1 = (i * side, j * side)
			p2 = ((i + 1) * side, (j + 1) * side)
			squares.append((p1, p2))
	return squares

def showImage(img):
	cv2.imshow('Grids', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

skip_dilate = False

original = cv2.imread("Sudoku.jpg", cv2.IMREAD_GRAYSCALE)

proc = cv2.GaussianBlur(original.copy(), (9, 9), 0)
proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
proc = cv2.bitwise_not(proc, proc)

if not skip_dilate:
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    proc = cv2.dilate(proc, kernel)

corners = findCorners(proc)
cropped = crop(original, corners)
squares = gridDetect(cropped)
print(squares)
displayRects(cropped, squares)

croppedImages = []

for x in range(0,81):
	croppedImages.append(cropped[int(squares[x][0][0]):int(squares[x][1][0]), int(squares[x][0][1]):int(squares[x][1][1])])

#croppedImages contains the 81 extracted images for use in MNIST!
#example
showImage(croppedImages[6])