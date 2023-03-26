import numpy as np

import torchvision.transforms as transforms
import torch

# 定义数据预处理的transform
transform = transforms.Compose([
    # 将PIL图像转换为PyTorch张量
    transforms.ToTensor(),
    # 归一化到[-1, 1]范围内
    # transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # 调整图像大小
    transforms.Resize((112, 112))
])

def get_score(period_score, within_period_score):
    """Combine the period and periodicity scores."""
    within_period_score = torch.sigmoid(within_period_score)[:, 0]
    per_frame_periods = torch.argmax(period_score, axis=-1) + 1
    pred_period_conf, _ = torch.max(torch.nn.functional.softmax(period_score, axis=-1), axis=-1)
    pred_period_conf  = torch.where(per_frame_periods < 3, 0.0, pred_period_conf)
    within_period_score *= pred_period_conf
    within_period_score = np.sqrt(within_period_score)
    pred_score = torch.mean(within_period_score)
    return pred_score, within_period_score

def get_counts(model, frames, strides, batch_size,
               threshold,
               within_period_threshold,
               constant_speed=False,
               median_filter=False,
               fully_periodic=False):
    """Pass frames through model and conver period predictions to count."""
    seq_len = len(frames)
    raw_scores_list = []
    scores = []
    within_period_scores_list = []    
    if fully_periodic:
        within_period_threshold = 0.0

    num_frames = 64
    image_size = 112
    frames = transform(frames)
    for stride in strides:
        num_batches = int(np.ceil(seq_len/num_frames/stride/batch_size))
        raw_scores_per_stride = []
        within_period_score_stride = []
        for batch_idx in range(num_batches):
            idxes = torch.arange(batch_idx * batch_size * num_frames* stride,
                      (batch_idx+1)* batch_size * num_frames * stride,
                      stride)
            
            idxes = torch.clamp(idxes, 0, seq_len-1)
            curr_frames = torch.index_select(frames, 0, idxes)

            curr_frames = curr_frames.view(batch_size, num_frames, image_size, image_size, 3)


            raw_scores, within_period_scores, _ = model(curr_frames)
            raw_scores_per_stride.append(np.reshape(raw_scores.numpy(),
                                                    [-1, num_frames//2]))
            within_period_score_stride.append(np.reshape(within_period_scores.numpy(),
                                                        [-1, 1]))
            
        raw_scores_per_stride = np.concatenate(raw_scores_per_stride, axis=0)
        raw_scores_list.append(raw_scores_per_stride)
        within_period_score_stride = np.concatenate(
            within_period_score_stride, axis=0)
        pred_score, within_period_score_stride = get_score(
            raw_scores_per_stride, within_period_score_stride)
        scores.append(pred_score)
        within_period_scores_list.append(within_period_score_stride)

    
  
    
