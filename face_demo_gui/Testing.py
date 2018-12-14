import os
import cPickle
from matplotlib import pyplot as plt
import PIL

dict_personimages = dict()

if os.path.exists('data/gui_dict.pickle'):
    with open('data/gui_dict.pickle','rb') as handle:
        dict_personimages = cPickle.load(handle)
        # print len(dict_personimages)
        # print dict_personimages
else:
    dict_personimages = dict()
    print "Didnt find file"


        # print key[0],key[1],key[2],key[3]

for i in range(len(dict_personimages['2 img'])):
    print dict_personimages['2 img'][i][0]



# print dict_personimages['D Fed'][0][0]
# plt.imshow(dict_personimages['D Fed'][0], interpolation='nearest')
# plt.show()



# img = PIL.Image.fromarray(dict_personimages['Dennis fed1image'][0])
# img.show()