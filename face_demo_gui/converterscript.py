import cPickle


with open('data/enroll_faces.pickle', 'rb') as handle:
    dict_old_images = cPickle.load(handle)

with open('data/enroll_feature.pickle', 'rb') as handle:
    dict_old_features = cPickle.load(handle)


dict_new_images = dict()
dict_new_features = dict()

for name in dict_old_images:
    frame0 = dict_old_images[name]['frontal_frame']
    rectangle0 = dict_old_images[name]['frontal_rectangles']
    frame1 = dict_old_images[name]['profile_frame']
    rectangle1 = dict_old_images[name]['profile_rectangles']

    dict_new_images[name] = {}
    temppair0 = [frame0,rectangle0]
    temppair1 = [frame1,rectangle1]
    list = [temppair0,temppair1]
    dict_new_images[name] = list

for name in dict_old_features:
    feat0 = dict_old_features[name]['frontal_feature']
    feat1 = dict_old_features[name]['profile_feature']
    dict_new_features[name] = {}
    dict_new_features[name] = [feat0, feat1]

with open('data/gui_dict.pickle', 'wb') as handle:
    cPickle.dump(dict_new_images, handle)

with open('data/gui_features.pickle', 'wb') as handle:
    cPickle.dump(dict_new_features, handle)