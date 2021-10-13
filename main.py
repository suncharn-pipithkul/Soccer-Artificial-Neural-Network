import numpy as np
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

y_train = np.array(
    [
        [ 5.00000000e+00,  5.00000000e+00,  8.40000000e+01,  1.05628826e+04],
        [ 6.00000000e+00,  3.00000000e+00,  1.93000000e+01, -3.16678922e+04],
        [ 8.00000000e+00,  4.00000000e+00,  1.13000000e+01,  7.15978449e+04]
    ]
)

y_test = np.array(
    [
        [3.0, 3.0, 2.0e+01, 2.0e+04],
        [4.0, 2.0, 2.0e+01, 9.0e+04]
    ]
)

def scale_down(y, ymin, ymax):
    return (y - ymin)/(ymax - ymin)

def scale_dataset_down(y_list):
    for exemplar in y_list:
        exemplar[0] = scale_down(exemplar[0], min_score, max_score)
        exemplar[1] = scale_down(exemplar[1], min_conceded, max_conceded)
        exemplar[2] = scale_down(exemplar[2], min_drink, max_drink)
        exemplar[3] = scale_down(exemplar[3], min_earn, max_earn)

def scale_up(y, ymin, ymax):
    return y * (ymax - ymin) + ymin

def scale_dataset_up(y_list):
    for exemplar in y_list:
        exemplar[0] = scale_up(exemplar[0], min_score, max_score)
        exemplar[1] = scale_up(exemplar[1], min_conceded, max_conceded)
        exemplar[2] = scale_up(exemplar[2], min_drink, max_drink)
        exemplar[3] = scale_up(exemplar[3], min_earn, max_earn)

y_all = np.concatenate((y_train, y_test))

max_score, max_conceded, max_drink, max_earn = np.max(y_all, axis=0)
min_score, min_conceded, min_drink, min_earn = np.min(y_all, axis=0)
print()

scale_dataset_down(y_train)

print()

scale_dataset_up(y_train)

print()
