import cv2
import xml.dom.minidom
import os
#img_path = '/home/acytoo/workSpace/paddle/object_detection/train/train/023b5bb5c9ea15ce0082c2b9bd003af33a87b215.jpg'

# img_info = '023b5bb5c9ea15ce0082c2b9bd003af33a87b215.jpg,30,231,97,664,224'.split(',')

# print(img_info, int(img_info[1]))
# img = cv2.imread(img_path)
# img = img[int(img_info[3]):int(img_info[5]),int(img_info[2]):int(img_info[4])]
# cv2.imshow('sd',img)
# cv2.waitKey(0)


def info_to_xml(each_line):

    # [023b5bb5c9ea15ce0082c2b9bd003af33a87b215.jpg,30,207,550,737,640]
    img_path = '../train/train/' + each_line[0]
    xml_path = '../train/annotation/'+each_line[0].split('.')[0] + '.xml'
    if os.path.exists(xml_path):
        print('exist')
        return  # for now, i will just ignore the image with two tags

    img = cv2.imread(img_path, 1)

    img_folder = 'images'
    img_filename = each_line[0]
    img_database = 'Unknown'
    img_width = str(len(img[0]))
    img_height = str(len(img))
    img_depth = str(len(img[0][0]))
    img_segmented = str(0)
    img_name = 'store_tag'
    img_pose = 'Unspecified'
    img_truncated = '0'
    img_difficult = str(0)
    img_xmin = each_line[2]
    img_ymin = each_line[3]
    img_xmax = each_line[4]
    img_ymax = each_line[5].replace('\n', '').replace('\r', '')

    doc = xml.dom.minidom.Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode(img_folder))
    root.appendChild(folder)

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(img_filename))
    root.appendChild(filename)

    path = doc.createElement('path')
    path.appendChild(doc.createTextNode(img_path))
    root.appendChild(path)

    source = doc.createElement('source')
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode(img_database))
    source.appendChild(database)
    root.appendChild(source)

    size = doc.createElement('size')
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(img_width))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(img_height))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(img_depth))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    root.appendChild(size)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode(img_segmented))
    root.appendChild(segmented)

    fileobject = doc.createElement('object')
    name = doc.createElement('name')
    name.appendChild(doc.createTextNode(img_name))
    pose = doc.createElement('pose')
    pose.appendChild(doc.createTextNode(img_pose))
    truncated = doc.createElement('truncated')
    truncated.appendChild(doc.createTextNode(img_truncated))
    difficult = doc.createElement('difficult')
    difficult.appendChild(doc.createTextNode(img_difficult))
    bndbox = doc.createElement('bndbox')
    xmin = doc.createElement('xmin')
    xmin.appendChild(doc.createTextNode(img_xmin))
    ymin = doc.createElement('ymin')
    ymin.appendChild(doc.createTextNode(img_ymin))
    xmax = doc.createElement('xmax')
    xmax.appendChild(doc.createTextNode(img_xmax))
    ymax = doc.createElement('ymax')
    ymax.appendChild(doc.createTextNode(img_ymax))
    bndbox.appendChild(xmin)
    bndbox.appendChild(ymin)
    bndbox.appendChild(xmax)
    bndbox.appendChild(ymax)
    fileobject.appendChild(name)
    fileobject.appendChild(pose)
    fileobject.appendChild(truncated)
    fileobject.appendChild(difficult)
    fileobject.appendChild(bndbox)
    root.appendChild(fileobject)

    fp = open(xml_path, 'w')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")


if __name__ == '__main__':
    txt_annotation = '../train/train.txt'
    annotation_file = open(txt_annotation, 'r')
    annotation_dir = '../train/annotation'
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)

    for each_line in annotation_file.readlines():
        info_to_xml(each_line.split(','))

    print('finish')
