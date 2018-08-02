"""
show the object detection result from infer.res

"""

import cv2


class part_object(obj_info):

    """
    Store rectangle info for each line, when there are multi-object in one img,
    draw rectangle for each of them
    """

    def __init__(self, obj_info):
        parts = obj_info.split(',')
        self.path = parts[0]
        self.obj_type = parts[1]
        self.prob = float(parts[2])
        self.xmin = float(parts[3])
        self.ymin = float(parts[4])
        self.xmax = float(parts[5])
        self.ymax = float(parts[6])

    def draw_rect(self, img):
        # img = cv2.imread(self.path, 1)
        cv2.rectangle(img, (int(self.xmin), int(self.ymin)),
                      (int(self.xmax), int(self.ymax)), (0, 255, 0), 3)
        # cv2.imshow('store_tag', img)
        # cv2.waitKey(25)
        return img

    def show_res(self):
        cv2.imshow('store_tag', self.img)
        cv2.waitKey(0)


if __name == '__main__':

    file = open('infer.res')
    last_obj_name = ''
    last_obj = None
    obj_num
    for each_line in file.readlines():
        current_obj = part_object(each_line)
        if last_obj_name.split(',')[0] == each_line.split(',')[0]:
            obj_num += 1

    file.close()
