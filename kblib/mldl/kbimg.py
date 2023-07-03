import cv2


def kbmosaic(img, rect, size):
    (x1, y1, x2, y2) = rect
    print, rect
    w = x2 - x1
    h = y2 - y1
    i_rect = img[y1:y2, x1:x2]

    i_small = cv2.resize(i_rect, (size, size))  # 얼굴 부분 그림 작게만들기
    # 다시 원래 사이즈로 늘리면 픽셀이 적은 상태에서 늘어나면서 이미지가 깨어져 보인다.
    i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)

    # plt.imshow(i_mos)

    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos

    return img2
