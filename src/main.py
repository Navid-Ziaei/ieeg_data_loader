from src.data.data_loader import iEEGDataLoader
from src.prepare_clear_data.prepare_data import CLEARDataLoader
from src.visualization import *
import ieeg_data_loader
# data_loader = CLEARDataLoader()
# data_loader.prepare_data()

patient = 'p01'
task = 'flicker'
path_result = ''
data_loader = iEEGDataLoader(patient=patient,
                             target_class='color',
                             prepared_dataset_path="D:\\Navid\\Dataset\\CLEAR\\interim\\",
                             task=task,
                             file_format='npy')
dataset_list = data_loader.load_data()
for dataset_idx, dataset in enumerate(dataset_list):
    print("Selected Block is {}".format(dataset_idx))

    dataset.get_time_annotation(patient=patient, task=task)

    trial_times = dataset.trial_time_annotation["trial end"] - dataset.trial_time_annotation["image onset"]
    # Define bin edges for the histogram
    continuous_signal, continuous_indicator = dataset.epoched_to_continuous(patient=patient, task=task,
                                                                            debug=False,
                                                                            save_path=path_result)

    plot_continuous_signal(continuous_signal, continuous_indicator,
                           channel_names=dataset.channel_name,
                           channel_number=15, task=task, save_path=None)
    plt.close()
    plt.cla()

    """plot_all_continuous_signal(continuous_signal, continuous_indicator,
                               channel_names=dataset.channel_name,
                               task=task, save_path='')
    plt.close()
    plt.cla()"""

    x, y, time = dataset.get_trial_data(patient, start_time=-500, end_time=2000)



