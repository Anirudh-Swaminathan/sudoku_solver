import cv2
import numpy as np
import operator

import keras
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.datasets import mnist

import np_utils


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
    cv2.waitKey(500)
    cv2.destroyAllWindows()


def findNumber(img, mod):
    """Function to recognize the number from the trained CNN

    @:param img - the small image to classify
    @:param mod - the model used for classifying the image
    @:returns - the integer that is detected
    """
    X = img.reshape(1, 28, 28, 1).astype('float32')
    ans = mod.predict(X, verbose=1)
    out = np.argmax(ans, axis=1)
    print ans
    showImage(img)
    if out == 0:
        out = int(raw_input("Check the image and input the integer:- "))
    return out


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
# cropped = crop(proc, corners)
squares = gridDetect(cropped)
# print(squares)
displayRects(cropped, squares)

croppedImages = []
numbered = list()

print "Displaying 81 images"
for x in range(0, 81):
    mini_img = cropped[int(squares[x][0][0]):int(squares[x][1][0]), int(squares[x][0][1]):int(squares[x][1][1])]
    m_img = cv2.resize(mini_img, (28, 28), interpolation=cv2.INTER_AREA)
    # m_img = mini_img[10:10+28, 6:6+28]
    blur = cv2.GaussianBlur(m_img, (5, 5), 0)
    # m_bin = cv2.adaptiveThreshold(m_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    m_bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Whitening the edges
    m_bin[0:8, :] = 255
    m_bin[:, 0:8] = 255

    # Invert the binary images
    m_bin = 255 - m_bin

    # m_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    croppedImages.append(m_bin)

    unique, counts = np.unique(m_bin, return_counts=True)
    print "For image " + str(x + 1) + " the unique and count is " + str(unique) + str(counts)
    if 784 - counts[0] < 65:
        print "Empty!"

    else:
        if len(counts) == 1 and unique[0] == 255:
            print "Empty!"
        else:
            print "Numbered:"
            ap = tuple((x, m_bin))
            numbered.append(ap)

print "There are ", len(numbered), " numbers in the grid"
# print len(croppedImages)
# print type(croppedImages[6])
# print croppedImages[6].shape
# print np.unique(croppedImages[6])

# croppedImages contains the 81 extracted images for use in MNIST!
# example
# showImage(croppedImages[6])
# for i in range(81):
#     showImage(croppedImages[i])

# Constructing the sudoku grid
grid = [["." for _ in range(9)] for _ in range(9)]
print len(grid), len(grid[0])

# Load the trained model for real-time classification
# mods = model_from_json(open('mnist.json').read())
# mods.load_weights('mnist_weights.h5')
mods = load_model('mnist.h5')
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_test = X_test / 255
y_test = keras.utils.to_categorical(y_test)
scores = mods.evaluate(X_test, y_test, verbose=1)
print scores
showImage(X_test[0])

for im in numbered:
    pos = tuple(((im[0] / 9), (im[0] % 9)))
    img = im[1]
    num = findNumber(img, mods)
    # showImage(img)
    # num = "n"
    grid[pos[0]][pos[1]] = num

for row in grid:
    for item in row:
        print item,
    print ""
