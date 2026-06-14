import torch
import torch.nn as nn

class Out_class(nn.Module):

    def __init__(self, d_model, class_num):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, class_num + 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x
    
    
class Out_score(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x).sigmoid()
        return x
    

def Get_Fighter_in_Loss(mask_logits):

    n, d = mask_logits.shape
    # [N, N]
    IOU_graph = get_iou_utiles(mask_logits)
    IOU_graph.fill_diagonal_(float('-inf'))
    # [N]
    IOU_indexs = torch.argmax(IOU_graph, dim = -1)
    # [N]
    Fighters_Mask = mask_logits[IOU_indexs]
    # [N]
    IOU = IOU_graph.gather(dim=-1, index=IOU_indexs.unsqueeze(1)).squeeze(1)

    return Fighters_Mask, IOU


def Get_Fighter_in_TS_with_index(mask_logits, class_logits, iouscore, mode = "times"):
    """
    Input :
    mask_logits [B, N, M]
    class_logits [B, N, 19]
    iouscore [B, N]
    Outout :
    Mask: [B, N, M]
    """
    Leaders = []
    Fighters = []
    b, n, d = mask_logits.shape
    class_logits_softmax =class_logits.softmax(dim = -1)
    for i in range(b):
        # [N, N]
        IOU_graph = get_iou_utiles(mask_logits[i])
        IOU_graph.fill_diagonal_(float('-inf'))
        
        # [N, N]
        if mode == "times":
            Score = get_score_with_iouscore(class_logits_softmax[i], iouscore[i])
        if mode == "plus":
            Score = get_score_with_iouscore_plus(class_logits_softmax[i], iouscore[i])
            
        # [N]
        IOU_indexs = torch.argmax(IOU_graph, dim = -1)
        # print("IOU_indexs shape is {}".format(IOU_indexs.shape))
        Fighters.append(IOU_indexs)
        # [N] 1 Leader -1 Loser
        Leader = Score.gather(dim=-1, index=IOU_indexs.unsqueeze(1)).squeeze(1)
        # print("Leader shape is {}".format(Leader.shape))
        Leaders.append(Leader)
    Leaders = torch.stack(Leaders)
    Fighters = torch.stack(Fighters)
    # print("Leaders shape is {}".format(Leaders.shape))
    # print("Fighters shape is {}".format(Fighters.shape))
    return Leaders, Fighters


def Get_Fighter_in_TS_with_index_SAMask(mask_logits, class_logits, iouscore, mode = "times"):
    """
    Input :
    mask_logits [B, N, M]
    class_logits [B, N, 19]
    iouscore [B, N]
    query [N, B, D]
    mode "times" / "plus" / "mix"
    SA_mode "normal" : Just use better information / "mix" : use better and worse information
    Outout :
    Mask: [B, N, N]
    """
    Leaders = []
    Fighters = []
    Masks = []
    
    # Score_Map = torch.bmm(query.transpose(0, 1).contiguous(), query.permute(1, 2, 0).contiguous()).sigmoid()
    # print(Score_Map.shape)
    b, n, _ = class_logits.shape
    class_logits_softmax =class_logits.softmax(dim = -1)

    
    for i in range(b):
        
        # [N, N]
        IOU_graph = get_iou_utiles(mask_logits[i])
        IOU_func_times = IOU_graph.clone()
        IOU_graph.fill_diagonal_(float('-inf'))
        
        # Leader: times
        # Attention: times
        if mode == "times":
            Score, Score_Leader, Score_worse = get_score_with_iouscore_NNgraph(class_logits_softmax[i], iouscore[i])
        # Leader: plus
        # Attention: plus
        if mode == "plus":
            Score, Score_Leader, Score_worse = get_score_with_iouscore_NNgraph_plus(class_logits_softmax[i], iouscore[i])
        # Leader: times
        # Attention: plus
        if mode == "mix":
            Score, Score_Leader, Score_worse = get_score_with_iouscore_NNgraph_NNtimes(class_logits_softmax[i], iouscore[i])
            
        # [N]
        IOU_indexs = torch.argmax(IOU_graph, dim = -1)
        Mask_add = (Score_Leader + Score_worse) 
        Masks.append(Mask_add * (1 - IOU_func_times))
        
        Fighters.append(IOU_indexs)
        # [N] 1 Leader -1 Loser
        Leader = Score.gather(dim=-1, index=IOU_indexs.unsqueeze(1)).squeeze(1)
        Leaders.append(Leader)
        
    Leaders = torch.stack(Leaders)
    Fighters = torch.stack(Fighters)
    Masks = torch.stack(Masks)
    return Leaders, Fighters, Masks


def Get_Fighter_in_TS_with_index_SAMask_Original(mask_logits, class_logits, iouscore, mode = "times"):
    """
    Input :
    mask_logits [B, N, M]
    class_logits [B, N, 19]
    iouscore [B, N]
    query [N, B, D]
    mode "times" / "plus" / "mix"
    SA_mode "normal" : Just use better information / "mix" : use better and worse information
    Outout :
    Mask: [B, N, N]
    """
    Leaders = []
    Fighters = []
    Masks = []
    
    # Score_Map = torch.bmm(query.transpose(0, 1).contiguous(), query.permute(1, 2, 0).contiguous()).sigmoid()
    # print(Score_Map.shape)
    b, n, _ = class_logits.shape
    class_logits_softmax =class_logits.softmax(dim = -1)[:, :, :-1]
    # print("original")
    for i in range(b):
        
        # [N, N]
        IOU_graph = get_iou_utiles(mask_logits[i])
        IOU_graph_ = IOU_graph.clone()
        IOU_graph.fill_diagonal_(float('-inf'))

        # IOU_50 = IOU_graph > 0.25
        
        # Leader: times
        # Attention: times
        if mode == "times":
            Score, _, _, score_difference = get_score_with_iouscore_NNgraph(class_logits_softmax[i], iouscore[i])
        # Leader: plus
        # Attention: plus
        if mode == "plus":
            Score, _, _, score_difference = get_score_with_iouscore_NNgraph_plus(class_logits_softmax[i], iouscore[i])
        # Leader: times
        # Attention: plus
        if mode == "mix":
            Score, _, _, score_difference = get_score_with_iouscore_NNgraph_NNtimes(class_logits_softmax[i], iouscore[i])
            
        # Leader&Fighter
        IOU_indexs = torch.argmax(IOU_graph, dim = -1)
        Fighters.append(IOU_indexs)
        # [N] 1 Leader -1 Loser
        Leader = Score.gather(dim=-1, index=IOU_indexs.unsqueeze(1)).squeeze(1)
        Leaders.append(Leader)
        
        # Masks
        # 计算Mask_add
        Mask_add = torch.where(score_difference >= 0, torch.ones_like(score_difference), torch.full_like(score_difference, -1))
        Mask_add *= IOU_graph_
        Masks.append(Mask_add)
        
    Leaders = torch.stack(Leaders)
    Fighters = torch.stack(Fighters)
    Masks = torch.stack(Masks)
    return Leaders, Fighters, Masks



def advanced_mapping(x):
    min_val_neg = torch.min(x[x < 0])
    x_pos = torch.exp(-(x**2)) * (x >= 0).float() 
    x_neg = (1 + torch.exp(-((2 + x/min_val_neg)**2))) * (x < 0).float() - 1
    return x_pos + x_neg




def get_iou_utiles(Mask: torch.Tensor):
    inputs = Mask.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    # binarized_inputs [N, M]
    # binarized_targets [N, M]
    # intersection [N]
    scores = []
    for i in range(0, Mask.shape[0], 40):
        inputs = binarized_inputs[i:i+40, :]
        binarized_targets = binarized_inputs
        intersection = (inputs.unsqueeze(1) * binarized_targets.unsqueeze(0)).sum(-1)
        union = inputs.unsqueeze(1).sum(-1) + binarized_targets.unsqueeze(0).sum(-1) - intersection
        # [10, M]
        score = intersection / (union + 1e-6)
        scores.append(score)
    scores = torch.concat(scores, dim = 0)
    return scores


def get_score_with_iouscore_NNgraph_plus(class_logits: torch.Tensor, iouscore: torch.Tensor):
    """
    Leader: plus
    Attention: plus
    """
    
    # max_score  [N]
    # iouscore [N]
    max_score = class_logits.max(dim = -1)[0] + iouscore
    # score_difference
    score_difference = max_score.unsqueeze(1) - max_score.unsqueeze(0)
    Leader = score_difference.clone()
    Leader[Leader > 0] = 1
    Leader[Leader < 0] = -1
    # Score_better [N, N] bool 
    Score_better = torch.where(score_difference > 0, score_difference, 0)
    Score_worse = torch.where(score_difference < 0, score_difference, 0)
    return Leader, Score_better, Score_worse, score_difference


def get_score_with_iouscore_NNgraph_NNtimes(class_logits: torch.Tensor, iouscore: torch.Tensor):
    """
    Leader: times
    Attention: plus
    """
    
    # max_score  [N]
    # iouscore [N]
    max_score = class_logits.max(dim = -1)[0] * iouscore
    # score_difference
    score_difference = max_score.unsqueeze(1) - max_score.unsqueeze(0)
    Leader = score_difference.clone()
    Leader[Leader > 0] = 1
    Leader[Leader < 0] = -1
    # Score_better [N, N] bool 
    max_score_plus = class_logits.max(dim = -1)[0] + iouscore
    score_difference_plus = max_score_plus.unsqueeze(1) - max_score_plus.unsqueeze(0)
    Score_better = torch.where(score_difference_plus > 0, score_difference_plus, 0) * (score_difference > 0)
    Score_worse = torch.where(score_difference_plus < 0, score_difference_plus, 0) * (score_difference < 0)
    return Leader, Score_better, Score_worse, score_difference



def get_score_with_iouscore_NNgraph(class_logits: torch.Tensor, iouscore: torch.Tensor):
    """
    Leader: times
    Attention: times
    """
    
    # max_score  [N]
    # iouscore [N]
    max_score = class_logits.max(dim = -1)[0] * iouscore
    # score_difference
    score_difference = max_score.unsqueeze(1) - max_score.unsqueeze(0)
    Leader = score_difference.clone()
    Leader[Leader > 0] = 1
    Leader[Leader < 0] = -1
    # Score_better [N, N] bool 
    Score_better = torch.where(score_difference > 0, score_difference, 0)
    Score_worse = torch.where(score_difference < 0, score_difference, 0)
    return Leader, Score_better, Score_worse, score_difference


def get_score_with_iouscore(class_logits: torch.Tensor, iouscore: torch.Tensor):
    """
    Leader: times
    """
    # max_score  [N]
    # iouscore [N]
    max_score = class_logits.max(dim = -1)[0] * iouscore
    # score_difference
    score_difference = max_score.unsqueeze(1) - max_score.unsqueeze(0)
    Leader = score_difference.clone()
    # print("score_difference shape is {}".format(score_difference.shape))
    Leader[Leader > 0] = 1
    Leader[Leader < 0] = -1
    return Leader


def get_score_with_iouscore_plus(class_logits: torch.Tensor, iouscore: torch.Tensor):
    """
    Leader: plus
    """
    # max_score  [N]
    # iouscore [N]
    max_score = class_logits.max(dim = -1)[0] + iouscore
    # score_difference
    score_difference = max_score.unsqueeze(1) - max_score.unsqueeze(0)
    Leader = score_difference.clone()
    # print("score_difference shape is {}".format(score_difference.shape))
    Leader[Leader > 0] = 1
    Leader[Leader < 0] = -1
    return Leader




def Get_expand_SORT_embeding(embedding, query, Index):
    """
    embedding: [N, D]
    query: [N, B, D]
    Index: [B, N]
    """
    n, b, d = query.shape
    # [N, D] -> [N, B, D] -> [B, N, D]
    embedding_expand = embedding.unsqueeze(1).repeat(1, b, 1).transpose(0, 1)
    # [B, N] -> [B, N, D]
    Index_expand = Index.unsqueeze(2).repeat(1, 1, d)
    # [B, N, D] -> [N, B, D]
    embedding_SORT = torch.gather(embedding_expand, 1, Index_expand).transpose(0, 1)
    return embedding_SORT

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [N, B, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.permute(1, 2, 0).contiguous()).permute(2, 0, 1).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError