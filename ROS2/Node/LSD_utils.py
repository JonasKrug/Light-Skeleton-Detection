import torch

def LSD_keypoint_nms(scores, device, prob_threshold):
    # We want to keep only the four highest keypoints that are connected (excluding the center rear light)
    # For reference:
    # ["Left Front Light", "Left Mirror", "Right Front Light", "Right Mirror", "Right Rear Light", "Center Rear Light", "Left Rear Light"]
    for i, keypoint_scores in enumerate(scores):
        view_matrix = torch.tensor([[1/4, 1/4, 1/4,   0,   0,   0, 1/4],
                                    [1/5, 1/5,   0,   0, 1/5, 1/5, 1/5],
                                    [  0, 1/5,   0, 1/5, 1/5, 1/5, 1/5],
                                    [  0,   0, 1/5, 1/5, 1/5, 1/5, 1/5],
                                    [1/4, 1/4, 1/4, 1/4,   0,   0,   0]])
        compensation = torch.tensor([4, 5, 5, 5, 4])
        view_matrix = view_matrix.to(device)
        vehicle_view_scores = torch.matmul(view_matrix, keypoint_scores)

        most_likely_view_index = torch.argmax(vehicle_view_scores)
        adjusted_scores = torch.mul(keypoint_scores, view_matrix[most_likely_view_index])
        adjusted_scores = torch.mul(adjusted_scores, compensation.to(device)[most_likely_view_index])

        adjusted_scores_prob = adjusted_scores/torch.max(adjusted_scores)
        adjusted_scores = torch.where(adjusted_scores_prob > prob_threshold, adjusted_scores, torch.zeros_like(adjusted_scores))
        
        scores[i] = adjusted_scores
    
    return scores
