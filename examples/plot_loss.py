import matplotlib.pyplot as plt
import pickle
import argparse


# dir = "examples/output/cell_inter2"
# # dir = f'/Users/brittany/Documents/GitHub/MORALS/examples/output/sphere2_500x/{ex_num}'
# train_log_file = dir + f'/logs/train_losses_0.pkl'
# test_log_file = dir + f'/logs/test_losses_0.pkl'


if __name__ == "__main__":

    config='cell_inter.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default=config)

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    with open(config_fname) as f:
        config = eval(f.read())


    dir = config['log_dir']
    train_log_file = dir + f'/train_losses_0.pkl'
    test_log_file = dir + f'/test_losses_0.pkl'
        
    for file in [train_log_file, test_log_file]:
        with open(file, 'rb') as f:
            logs = pickle.load(f)

        print(logs.keys())

        plt.plot(logs['loss_ae1'], label = 'loss_ae1')
        plt.plot(logs['loss_ae2'], label = 'loss_ae2')
        plt.plot(logs['loss_dyn'], label = 'loss_dyn')
    # plt.plot(logs['loss_topo'], label = 'loss_topo')
        plt.plot(logs['loss_total'], label = 'loss_total')
        plt.legend()

        # make log scale 
        #plt.yscale('log')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.savefig(dir + f'losses_{file.split("/")[-1].split(".")[0]}.png', bbox_inches='tight')

        # plt.show()
        plt.close()