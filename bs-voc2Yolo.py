import xml.etree.ElementTree as ET
import os





'''
把labelimg标好的数据转化为yolo能用的版本
输入存放voc标签文件夹路径
输出转换好的标签到文件夹
'''

# voc_path = "dataSet/map1-h7-y10-Annotations"
voc_paths = [
    "F:/学习/2021-05ICRA/树莓派小车车/realsense/realsense-rover-A",
]

txt_path = "F:/学习/2021-05ICRA/树莓派小车车/realsense/realsense-rover-yolo"
classes = ["rover"]


for voc_path in voc_paths:
    xmlL = os.listdir(voc_path)
    for xml_file in xmlL:
        out_file = os.path.splitext(os.path.basename(xml_file))[0]+".txt"
        out_file_absolute = os.path.join(txt_path,out_file)
        fo = open(out_file_absolute,"w")

        xml_file_absolute = os.path.join(voc_path,xml_file)

        in_file = open(xml_file_absolute, encoding='utf-8')
        tree=ET.parse(in_file)
        root = tree.getroot()


        w = int(root.find("size").find("width").text)
        h = int(root.find("size").find("height").text)


        for obj in root.iter('object'):
            category = classes.index(obj.find('name').text)
            xmlbox = obj.find('bndbox')
            b = (
                    (float(xmlbox.find('xmax').text)+float(xmlbox.find('xmin').text))/2/w, 
                    (float(xmlbox.find('ymax').text)+float(xmlbox.find('ymin').text))/2/h,
                    (float(xmlbox.find('xmax').text)-float(xmlbox.find('xmin').text))/w,
                    (float(xmlbox.find('ymax').text)-float(xmlbox.find('ymin').text))/h,
                )

            fo.write("{} {} {} {} {}".format(category,*b))
            fo.write("\r")
        fo.close()