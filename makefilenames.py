import os
import pickle
conferences = ['CVPR2013', 'CVPR2014', 'CVPR2015', 'CVPR2016', 'CVPR2017', 'CVPR2018', 'CVPR2019',
               'ICCV2013', 'ICCV2015', 'ICCV2017', 'ICCV2019', 'ECCV2018']


def searchfor(filename, folder):
    filelist = os.listdir(folder)
    for i in filelist:
        if i == filename:
            return True
    return False


if __name__ == '__main__':
    filenames = []
    for conf in conferences:
        image_filenames = os.listdir('D:/image_extraction/image_extraction_'+conf)
        for filename in image_filenames:
            if searchfor(filename, 'D:/caption_extraction/'+conf+'_extraction'):
                filenames.append(conf+'/'+filename)
    print(len(filenames))
    print(filenames)
    with open('filenames.pickle', 'wb') as f:
        pickle.dump(filenames, f)
    with open('filenames.pickle', 'rb') as r:
        data = pickle.load(r)
        print(len(data))
        print(data)
