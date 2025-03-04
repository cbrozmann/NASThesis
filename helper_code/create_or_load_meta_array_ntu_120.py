import numpy as np

# create meta_file
#
classes_list = ["drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down", "standing up (from sitting position)", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping (one foot jumping)", "jump up", "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger", "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person", "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person", "hugging other person", "giving something to other person", "touch other person's pocket", "handshaking", "walking towards each other", "walking apart from each other", "put on headphone", "take off headphone", "shoot at the basket", "bounce ball", "tennis bat swing", "juggling table tennis balls", "hush (quite)", "flick hair", "thumb up", "thumb down", "make ok sign", "make victory sign", "staple book", "counting money", "cutting nails", "cutting paper (using scissors)", "snapping fingers", "open bottle", "sniff (smell)", "squat down", "toss a coin", "fold paper", "ball up paper", "play magic cube", "apply cream on face", "apply cream on hand back", "put on bag", "take off bag", "put something into a bag", "take something out of a bag", "open a box", "move heavy objects", "shake fist", "throw up cap/hat", "hands up (both hands)", "cross arms", "arm circles", "arm swings", "running on the spot", "butt kicks (kick backward)", "cross toe touch", "side kick", "yawn", "stretch oneself", "blow nose", "hit other person with something", "wield knife towards other person", "knock over other person (hit with body)", "grab other person’s stuff", "shoot at other person with a gun", "step on foot", "high-five", "cheers and drink", "carry something with other person", "take a photo of other person", "follow other person", "whisper in other person’s ear", "exchange things with other person", "support somebody with hand", "finger-guessing game (playing rock-paper-scissors)"]
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
