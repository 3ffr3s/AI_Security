from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2, torch
from utils import get_preds

def returnCAM(feature_map, weight_softmax, class_idx, image_size):
    size_upsample = (image_size, image_size)
    bz, c, h, w = feature_map.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_map.reshape(c,h*w))
        cam = cam.reshape(h, w)
        cam -= np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255*cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam(net, feature_blobs, image_data, dataset_name):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    num_data = image_data.shape[0]
    image_size = image_data[0].shape[-1]
    CCI_CAM = torch.zeros((num_data, image_size, image_size))
    
    for i in range(num_data):
        preds, probs = get_preds(net, image_data[i].unsqueeze(0), dataset_name)
        CAMs = returnCAM(feature_blobs[i], weight_softmax, [preds], image_size)
    
        img = np.array(image_data[i].permute(1,2,0))
        height, width, _ = img.shape
        CAM = cv2.resize(CAMs[0], (width, height))
        CCI_CAM[i] = torch.from_numpy(CAM)
#         heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
#         img_var = np.uint8(255*img)         
#         result = heatmap * 0.3 + img_var * 0.5
#         plt.imshow(np.uint8(result))     
#         plt.show()
    return CCI_CAM