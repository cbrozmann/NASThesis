import numpy as np

# create meta_file
#
classes_list = ["drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down", "standing up (from sitting position)", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping (one foot jumping)", "jump up", "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger", "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person", "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person", "hugging other person", "giving something to other person", "touch other person's pocket", "handshaking", "walking towards each other", "walking apart from each other"]
arr = np.array(classes_list)
with open('meta_classes.npy', "wb") as f:
    np.save(f, arr)

# load meta_file
#
# classes_array = []
# test_array = np.load('meta_classes.npy')
#
# print(test_array)
# print(len(test_array))
# print(test_array[2])
