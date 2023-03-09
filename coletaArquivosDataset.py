import os

diretorio = 'dataset'
image_list = []
for root, subdirs, files in os.walk(diretorio):
    #cont = 0
    for file in files:
        if file.endswith('.jpg'):
            image_list.append(str(os.path.join(root, file)))
            #cont = 1

