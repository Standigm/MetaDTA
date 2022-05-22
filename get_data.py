import os
import wget



def data_download():
    # test_coo.pkl
    data_path = './data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(os.path.join(data_path,'test_coo.pkl')):
        filename = wget.download('https://figshare.com/ndownloader/files/35214781', data_path)
        assert(filename == os.path.join(data_path,'test_coo.pkl'))
    else:
        print('test_coo.pkl exists')

    # train_coo.pkl
    if not os.path.exists(os.path.join(data_path,'train_coo.pkl')):
        filename = wget.download('https://figshare.com/ndownloader/files/35215216', data_path)
        assert(filename == os.path.join(data_path,'train_coo.pkl'))
    else:
        print('train_coo.pkl exists')

    # total_ecfp.npy
    if not os.path.exists(os.path.join(data_path, 'total_ecfp.npy')):
        filename = wget.download('https://figshare.com/ndownloader/files/35215219', data_path)
        assert(filename == os.path.join(data_path, 'total_ecfp.npy'))
    else:
        print('total_ecfp.npy exists')

if __name__ == '__main__':
    data_download()
