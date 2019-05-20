import numpy as np
import pywt


def get_wavelet(image):
    dwt = pywt.dwt2(image, 'haar')
    wavelet = np.vstack([dwt[0], dwt[1][0], dwt[1][1], dwt[1][2]])
    return wavelet

def get_spatial(wavelet):
    size = wavelet.shape[2]
    one_size = size / 4
    wavelet_tuple = (wavelet[:,:,:one_size], [wavelet[:,:,one_size:one_size*2] , wavelet[:,:,one_size*2:one_size*3], wavelet[:,:,one_size*3:]])
    spatial = pywt.idwt2(wavelet_tuple, 'haar', axes=(0, 1))
    return spatial

def to_np(tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.squeeze(tensor)
    if tensor.ndim == 3:
        tensor = np.rollaxis(tensor, 0, 3)
    return tensor

def get_model_dir(scale, wavelets, l_channel, y_channel=False):
    if (y_channel):
        return "%d-%s-%s" % (scale, "wavelets" if wavelets else "spatial", "Y")
    return "%d-%s-%s" % (scale, "wavelets" if wavelets else "spatial", "L" if l_channel else "RGB")

BEST_EPOCHS = {
    get_model_dir(2, False, False): 30,
    get_model_dir(2, False, True): 20,
    get_model_dir(2, True, False): 28,
    get_model_dir(2, True, True): 32,
    get_model_dir(3, False, False): 37,
    get_model_dir(3, False, True): 34,
    get_model_dir(3, True, False): 33,
    get_model_dir(3, True, True): 35,
    get_model_dir(4, False, False): 33,
    get_model_dir(4, False, True): 31,
    get_model_dir(4, True, False): 19,
    get_model_dir(4, True, True): 12,
}
