import cv2

"""
[path, type, confidence, xmin, ymin, xmax, ymax]
"""
last_obj_info = ''
curr_obj_info = ''
obj_num = 0
rects = []


def obj_iter(res_file_path):
    for each_obj in open(res_file_path, 'r').readlines():
        yield each_obj


obj_iterator = obj_iter('infer.res')
curr_obj_info = next(obj_iterator).replace('\t', ' ').replace('\n', ' ')
curr_parts = curr_obj_info.split(' ')
next_obj_info = next(obj_iterator).replace(
    '\t', ' ').replace('\n', ' ')
next_parts = next_obj_info.split(' ')

while True:
    try:
        # print(curr_parts)
        img1 = cv2.imread(curr_parts[0], 1)
        img1 = cv2.rectangle(img1, (int(float(curr_parts[3])), int(float(curr_parts[4]))), (int(
            float(curr_parts[5])), int(float(curr_parts[6]))), (255, 137, 0), 3)

        # print(next_parts)
        while curr_parts[0] == next_parts[0]:
            img1 = cv2.rectangle(img1, (int(float(next_parts[3])), int(float(next_parts[4]))), (int(
                float(next_parts[5])), int(float(next_parts[6]))), (255, 137, 0), 3)
            current_obj_info = next_obj_info
            curr_parts = next_parts.copy()
            next_obj_info = next(obj_iterator).replace(
                '\t', ' ').replace('\n', ' ')
            next_parts = next_obj_info.split(' ')
            continue

        cv2.imshow('store_tag', img1)
        current_obj_info = next_obj_info
        curr_parts = next_parts.copy()
        next_obj_info = next(obj_iterator).replace(
            '\t', ' ').replace('\n', ' ')
        next_parts = next_obj_info.split(' ')
        if cv2.waitKey(0) & 0xff == ord('q'):
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        curr_obj_info = next(obj_iterator).replace(
            '\t', ' ').replace('\n', ' ')
