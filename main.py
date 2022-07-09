from modeling.smqFusion.smqFusion import smqFusion
import torch
from torchinfo import summary
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    model = smqFusion('resnet50',num_classes=3)
    print(model)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
