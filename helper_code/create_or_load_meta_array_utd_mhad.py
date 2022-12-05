import numpy as np

# create meta_file
#
classes_list = ['swipe_left', 'swipe_right', 'wave', 'clap', 'throw', 'arm_cross', 'basketball_shoot', 'draw_x', 'draw_circle_CW', 'draw_circle_CCW', 'draw_triangle', 'bowling', 'boxing', 'baseball_swing', 'tennis_swing', 'arm_curl', 'tennis_serve', 'push', 'knock', 'catch', 'pickup_throw', 'jog', 'walk', 'sit2stand', 'stand2sit', 'lunge', 'squat']
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
