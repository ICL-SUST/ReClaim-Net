import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
from trainers import trainer, reclaim_net_train
from datasets import dataloaders
from models.reclaim_net import ReClaimNet


args = trainer.train_parser()
with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
fewshot_path = os.path.join(data_path,'CUB_fewshot_cropped')

pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)

train_way = args.train_way
shots = [args.train_shot, args.train_query_shot]

train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                way=train_way,
                                                shots=shots,
                                                transform_type=args.train_transform_type)

model = ReClaimNet(way=train_way,
            shots=[args.train_shot, args.train_query_shot],
            resnet=args.resnet)

train_func = partial(reclaim_net_train.default_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)


#
# import torch
# from thop import profile, clever_format
# from models.reclaim_net import ReClaimNet
# from models.FRN import FRN
# from utils import util
#
# # 设置 GPU 和模型路径
# gpu = 0
# torch.cuda.set_device(gpu)
# model_path = '/code/zwy/Bi-FRN-main/experiments/CUB_fewshot_cropped/FRN/ResNet-12/wode3ResNet-12.pth'
#
# # few-shot 参数（根据你的日志）
# way = 10
# shot = 5
# query_shot = 15
# total_imgs = way * (shot + query_shot)
#
# # 初始化模型
# model = FRN(resnet=True, way=way, shots=[shot, query_shot])
#
# # 先包裹（如果模型内部未处理 DataParallel）
# # model = torch.nn.DataParallel(model)
#
# # 再统一移动到 GPU
# model = model.cuda()
# def clean_state_dict(state_dict):
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith("feature_extractor.module.module."):
#             new_key = k.replace("feature_extractor.module.module.", "feature_extractor.")
#         elif k.startswith("feature_extractor.module."):
#             new_key = k.replace("feature_extractor.module.", "feature_extractor.")
#         else:
#             new_key = k
#         new_state_dict[new_key] = v
#     return new_state_dict
#
# # 加载权重后不再 .cuda()
# state_dict = torch.load(model_path, map_location=util.get_device_map(gpu))
# # 清理键名
# state_dict = clean_state_dict(state_dict)
#
# # 加载到模型
# model.load_state_dict(state_dict, strict=True)
# for name, param in model.named_parameters():
#     if param.device.type != 'cuda':
#         print(f"[❗] PARAM NOT ON GPU: {name} is on {param.device}")
#
# model.eval()
#
# # 设置推理所需属性
# model.ways = way
# model.shots = [shot, query_shot]
#
# # 使用 THOP 分析 FLOPs 和参数
# dummy_input = torch.randn(total_imgs, 3, 84, 84).cuda()
# macs, params = profile(model, inputs=(dummy_input,))
# macs, params = clever_format([macs, params], '%.3f')
#
# print(f"[thop] FLOPs: {macs}")
# print(f"[thop] Params: {params}")
# # import time
# # import torch
# # from models.reclaim_net import ReClaimNet
# #
# # # 模型参数（与你训练一致）
# # way = 15
# # shot = 5
# # query_shot = 15
# # img_size = (3, 84, 84)
# # batch_size = way * (shot + query_shot)
# #
# # # 初始化模型
# # model = ReClaimNet(resnet=True, way=way, shots=[shot, query_shot])
# # model = model.cuda()
# # model.train()
# #
# # # 构造 dummy 输入和目标（假设是分类）
# # x = torch.randn(batch_size, *img_size).cuda()
# # y = torch.randint(0, way, (way * query_shot,)).cuda()  # 假标签
# #
# # # 优化器 & 损失函数
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# # criterion = torch.nn.NLLLoss()
# #
# # # 测多次求平均
# # trials = 1000
# # start = time.time()
# # for _ in range(trials):
# #     optimizer.zero_grad()
# #     out = model(x)                  # forward
# #     loss = criterion(out, y)       # loss
# #     loss.backward()                # backward
# #     optimizer.step()               # update
# # torch.cuda.synchronize()
# # end = time.time()
# #
# # avg_time = (end - start) / trials
# # print(f"[Training] Avg time per batch: {avg_time:.3f}s")