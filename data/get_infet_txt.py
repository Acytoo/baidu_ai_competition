"""
get text for infer from train.txt

"""


file = open('test.txt', 'r')
infer_file = open('infer.txt', 'w')
for each_line in file.readlines():
    infer_path = each_line.split(' ')[0]
    print(infer_path)
    infer_file.writelines(infer_path+'\n')


file.close()
infer_file.close()
