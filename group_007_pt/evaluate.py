import numpy as np
import torch
import model_UNet
from data_into_loaders import get_data
from utils import evaluate_model


def evaluate_all_models():
    # hyper params to create the valid and test data sets
    supervised_pct=0.05
    val_pct = 0.2
    test_pct = 0.1
    batch_size = 32
    img_resize = 64


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model = model_UNet.UNet(in_channels=3, num_classes=2, depth=3)
    model.to(device)


    print('\nLOADING DATALOADER - can take a 1-5 mins to download and process data')
    print('Need dataloader to evaluate the models on the test dataset')
    _, _, test_loader = get_data(supervised_pct,1 - supervised_pct, val_pct, test_pct, batch_size=batch_size, img_resize=img_resize)
    print('DATALOADER CREATION COMPLETE')

    model_paths = [
        'models/M05L/model_best_epoch_81.pt',
        'models/M05/model_best_epoch_61.pt',
        'models/M10L/model_best_epoch_66.pt',
        'models/M10/model_best_epoch_71.pt',
        'models/M25L/model_best_epoch_46.pt',
        'models/M25/model_best_epoch_76.pt',
        'models/M100L/model_best_epoch_46.pt'
    ]
    model_name_dict = {'M05L':0, 'M05':1, 'M10L':2, 'M10':3, 'M25L':4, 'M25':5, 'M100L':6}
    model_best_dict = {'M05L':81, 'M05':61, 'M10L':66, 'M10':71, 'M25L':46, 'M25':76, 'M100L':46}



    nb_models = len(model_name_dict)
    running_losses = np.zeros(shape=(nb_models, 100))
    sup_losses = np.zeros(shape=running_losses.shape)
    unsup_losses = np.zeros(shape=running_losses.shape)
    weights = np.zeros(shape=running_losses.shape)
    val_ious = np.zeros(shape=(nb_models, 20))
    val_accs = np.zeros(shape=val_ious.shape)
    test_accs = np.zeros(nb_models)
    test_ious = np.zeros(nb_models)

    logs = [
        running_losses,
        sup_losses,
        unsup_losses,
        weights,
        val_ious,
        val_accs,
        test_accs,
        test_ious
    ]

    log_file_names = [
        'running_loss',
        'sup_loss',
        'unsup_loss',
        'weights',
        'val_IOU',
        'val_accuracy',
        '',
        '',
    ]

    # Load the evaluation metrics that were logged during training:
    for model_id, model_name in enumerate(model_name_dict):
        folder_path = 'models/' + model_name + '/'
        for metric_id, log_file_name in enumerate(log_file_names):
            file_path = folder_path + log_file_name
            try:
                model_metric = np.loadtxt(file_path)
                n = model_metric.shape[0]
                logs[metric_id][model_id,:n] = model_metric
            except:
                pass
    
    

    # Function to print metric logged every epoch to screen
    def print_table_metric(metrics):
        print(f'{"Epoch":<8}|{" M05L":<8}|{" M05":<8}|{" M10L":<8}|{" M10":<8}|{" M25L":<8}|{" M25":<8}|{" M100L":<8}')
        line_string = '-'*70
        print(line_string)
        for epoch in np.arange(90):
            print(f'{epoch+1:8d}',end='')
            for model_id in np.arange(len(model_name_dict)):
                metric = metrics[model_id][epoch]
                print(f'|{metric:8.1f}', end='')
            print('')

    # Function to print metric logged every 5 epochs to screen
    def print_table_metric_percent(metrics):
        print(f'{"Epoch":<8}|{" M05L":<8}|{" M05":<8}|{" M10L":<8}|{" M10":<8}|{" M25L":<8}|{" M25":<8}|{" M100L":<8}')
        line_string = '-'*70
        print(line_string)
        for epoch_id in np.arange(20):
            print(f'{epoch_id * 5 + 1:8d}',end='')
            for model_id in np.arange(len(model_name_dict)):
                metric = metrics[model_id][epoch_id]
                print(f'|{metric:8.1%}', end='')
            print('')
    print("{:<20} {:<10} {:<10}".format("Model Name", "Test Accuracy", "Test IOU"))


    # print logged metrics:
    print('\nModel Losses\n***********************')
    print_table_metric(running_losses)
    print('\nValidation Accuracy\n***********************')
    print_table_metric_percent(val_accs)
    print('\nValidation IOU\n***********************')
    print_table_metric_percent(val_ious)
    
    
    # Calculate test evaluation metrics
    print('\nCalculating Test Evaluation Metrics\n***********************')
    for model_id, model_name in enumerate(model_name_dict):
        model_path = model_paths[model_id]
        
        # load model onto the cpu then move to device
        model.to('cpu')
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        
        # calculate test evaluation metrics
        acc, iou = evaluate_model(model, test_loader, device)
        test_accs[model_id] = acc
        test_ious[model_id] = iou
        print(f'Completed {model_name} Test Evaluation...')
    print('Completed calculating Test evaluation metrics')


    # Test Metrics on Best Performing Model
    print('\nTest IOU on Best Performing Model on Validation Set IOU\n***********************')
    print("{:<20}|{:<15}|{:<15}".format("Model Name", " Test Accuracy", " Test IOU"))
    line_string = '-'*50
    print(line_string)
    for id, model_name in enumerate(model_name_dict):
        best_epoch_id = model_best_dict[model_name] - 1
        print(f'{model_name:<20}|{test_accs[id]:15.1%}|{test_ious[id]:15.1%}')