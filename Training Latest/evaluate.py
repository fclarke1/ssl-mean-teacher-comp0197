import os
import torch
from data_into_loaders import get_data
from utils import evaluate_model
from model_UNet import UNet


def print_Table(dictionary, remove = None):
    '''Print the dictionary as a table'''
    #remove the keys that we don not want to print
    if remove:
        for item in remove:
            dictionary.pop(item)
    col_size = []
    for item in dictionary.keys():
        length = len(item)
        for i in range(len(dictionary[item])):
            if type(dictionary[item][i]) == str:
                if len(dictionary[item][i]) > length:
                    length = len(dictionary[item][i])
            else:
                if len(f'{dictionary[item][i]:.2%}') > length:
                    length = len(f'{dictionary[item][i]:.2%}')
        col_size.append(length + 2)
    for i in range(len(dictionary.keys()) - 1):
        print(list(dictionary.keys())[i], end = ' ' * (col_size[i] - len(list(dictionary.keys())[i])) + '|')
    print(list(dictionary.keys())[i + 1])
    print('-' *sum(col_size))
    for i in range(len(dictionary[item])):
        for j in range(len(dictionary.keys()) - 1):
            if type(dictionary[list(dictionary.keys())[j]][i]) != str:
                print(f'{dictionary[list(dictionary.keys())[j]][i]:.2%}', end = ' ' * (col_size[j] - len(f'{dictionary[list(dictionary.keys())[j]][i]:.2%}')) + '|')
            else:
                print(dictionary[list(dictionary.keys())[j]][i], end = ' ' * (col_size[j] - len(dictionary[list(dictionary.keys())[j]][i])) + '|')
        if type(dictionary[list(dictionary.keys())[j]][i]) != str:
            print(f'{dictionary[list(dictionary.keys())[j + 1]][i]:.2%}')
        else:
            print(dictionary[list(dictionary.keys())[j + 1]][i])
def evaluate():
    supervised_pct = 0.25
    val_pct, test_pct = 0.2, 0.1
    batch_size = 32
    img_resize = 64
    depth = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = UNet(in_channels = 3, num_classes = 2, depth = depth)
    loder_path = os.path.abspath(os.curdir)
    _, _, test_loader = get_data(supervised_pct,1 - supervised_pct, val_pct, test_pct, batch_size=batch_size, img_resize=img_resize)
    # test_loader = torch.rand(32,3,64,64)
    dictionary = {'Case':[],'Accuracy':[],'IOU':[],'Model':[]}
    files_list=os.listdir()
    for file in files_list:
        if file.find('.') == -1 and file != 'data' and file != '__pycache__':
            os.chdir(file)
            models_list = os.listdir()
            for model in models_list:
                os.chdir(model)
                list_of_f = os.listdir()
                for f in list_of_f:
                    if f.find('best') > 0:
                        local_path = os.path.abspath(os.curdir)
                        network.load_state_dict(torch.load(f, map_location = device))
                        os.chdir(loder_path)
                        acc, IOU = evaluate_model(network, test_loader, device)
                        # acc, IOU = 0.5, 0.8
                        os.chdir(local_path)
                        dictionary['Case'].append(model)
                        dictionary['Model'].append(f)
                        dictionary['Accuracy'].append(acc)
                        dictionary['IOU'].append(IOU)
                        break
                os.chdir('..')
            os.chdir('..')
    # print(dictionary)
    print_Table(dictionary,['Model'])
                

if __name__ == '__main__':
    evaluate()