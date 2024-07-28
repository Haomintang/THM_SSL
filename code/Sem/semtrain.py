
from Semdataset import *
#from Semshangdataset import *
from Vnet import VNet
import torch.optim as optim
from monai.losses import DiceLoss
from medpy import metric
from matplotlib import pyplot as plt
import argparse
import matplotlib

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def HD(pred, gt):
    hd = metric.binary.hd(pred, gt)
    return hd

def asd(pred, gt):
    asd = metric.binary.asd(pred, gt)
    return asd

def Dice(inp, target, eps=1):

    input_flatten = inp.flatten()  # 抹平了，弄成一维的
    target_flatten = target.flatten()
    # 计算交集中的数量
    overlap = np.sum(input_flatten * target_flatten)#相交的地方
    # print(target_flatten)
    # print(np.sum(target_flatten))
    # print(np.sum(input_flatten))
    # 返回值，让值在0和1之间波动
    return np.clip(((2 * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)

def ppv(inp, target):

    input_flatten = inp.flatten()  # 抹平了，弄成一维的
    target_flatten = target.flatten()
    # 计算交集中的数量
    overlap = np.sum(input_flatten * target_flatten)#相交的地方
    A = np.sum(input_flatten)

    return overlap / A

def sensitivity(inp, target):

    input_flatten = inp.flatten()  # 抹平了，弄成一维的
    target_flatten = target.flatten()
    # 计算交集中的数量
    overlap = np.sum(input_flatten * target_flatten)#相交的地方
    B = np.sum(target_flatten)

    return overlap / B


test_images = os.listdir('E:/fei EZF CT/3DUnet/dataset_xia/Semi/test/images')
test_labels = os.listdir('E:/fei EZF CT/3DUnet/dataset_xia/Semi/test/labels')


test_image_list = []
test_label_list = []

for i in test_images:
    file_path = 'E:/fei EZF CT/3DUnet/dataset_xia/Semi/test/images/{}'.format(i)
    #print(file_path)
    test_image_list.append(file_path)

for i in test_labels:
    file_path = 'E:/fei EZF CT/3DUnet/dataset_xia/Semi/test/labels/{}'.format(i)
    #print(file_path)
    test_label_list.append(file_path)

images = os.listdir('E:/fei EZF CT/3DUnet/dataset_xia/Semi/train/images')
labels = os.listdir('E:/fei EZF CT/3DUnet/dataset_xia/Semi/train/labels')

image_list = []
label_list = []

for i in images:
    file_path = 'E:/fei EZF CT/3DUnet/dataset_xia/Semi/train/images/{}'.format(i)
    # print(file_path)
    image_list.append(file_path)

for i in labels:
    file_path = 'E:/fei EZF CT/3DUnet/dataset_xia/Semi/train/labels/{}'.format(i)
    # print(file_path)
    label_list.append(file_path)

labelimage_list = list(range(19)) # (10-20)
unlabelimage_list = list(range(19,40))

batch_sampler = TwoStreamBatchSampler(labelimage_list, unlabelimage_list, 3, 2)


train_ds = D3VnetData(image_list, label_list)
train_loader = DataLoader(train_ds, batch_sampler=batch_sampler)

test_ds = D3VnetData_test(test_image_list , test_label_list)#看dataset里写的image_list
testloader = DataLoader(test_ds, batch_size=2,shuffle=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def create_model(ema=False):
    # Network definition
    net = VNet(1,2)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()  # 切断反向传播
    return model
model = create_model().to(device)
ema_model = create_model(ema=True).to(device)

optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001,betas=(0.9,0.99), weight_decay=0.0001)

labeled_bs = 1
max_iterations = 2000
consistency = 0.1
consistency_rampup = 40


parser = argparse.ArgumentParser()

if __name__ == '__main__':
    train_loss = []
    train_dice = []
    a = 0
    iter_num = 0
    for i in range(80):
        a = a + 1
        loss_train = 0
        dice = 0
        ASD = 0
        PPV = 0
        sen = 0
        hd = 0

        for i_batch, sampled_batch in enumerate(train_loader):


            volume_batch, label_batch = sampled_batch
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            unlabeled_volume_batch = volume_batch[1:]  # 无标签是后面的

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)  # 加入噪声
            ema_inputs = unlabeled_volume_batch + noise
            print(volume_batch.shape)
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)  # 加噪结果
                ema_output_soft = torch.softmax(ema_output, dim=1)

            T = 1#8
            volume_batch_r = volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, 132, 160, 160]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, 132, 160, 160)
            preds = torch.mean(preds, dim=0)  # (batch, 2, 112,112,80)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1,
                                           keepdim=True)  # (batch, 1, 112,112,80)

            print(a)
            # if a == 90:
            #     uncertainty0 = uncertainty.squeeze()
            #     uncertainty0 = uncertainty0[:1, :, :, :, ]
            #     uncertainty0 = uncertainty0.squeeze()
            #     uncertainty0 = uncertainty0[10, :, :]
            #
            #     uncertainty0 = uncertainty0.detach().cpu().numpy()
            #     #uncertainty0  = uncertainty0[::-1]
            #     #uncertainty0[uncertainty0  < 0.6] = 0.1
            #
            #     # plt.imshow(uncertainty0, cmap='gray')
            #     # plt.show()
            #
            #     max_index = np.unravel_index(np.argmax(uncertainty0, axis=None), uncertainty0.shape)
            #     max_value = uncertainty0[max_index]
            #     print(max_index, max_value)
            #     # norm = matplotlib.colors.Normalize(vmin=0, vmax = 0.69314)
            #     max_value = round(max_value, 3)
            #
            #     smc = plt.imshow(uncertainty0, cmap='jet', vmin=0)
            #     cb = plt.colorbar(smc, fraction=0.046, pad=0.04, shrink=1.0)
            #     cb.set_ticks([0, max_value])
            #
            #
            #     #plt.savefig('E:/fei EZF CT/3DUnet/Semi_/uncertaintymap.png', dpi=500)#40
            #     #plt.savefig('E:/fei EZF CT/3DUnet/Semi_/uncertaintymap80.png', dpi=500)  # 80
            #     plt.savefig('E:/fei EZF CT/3DUnet/Semi_/uncertaintymap90shang.png', dpi=500)  # 110
            #     plt.show()

            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs].long())
            loss_seg_dice = dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            consistency_dist = torch.mean((outputs_soft[labeled_bs:] - ema_output_soft) ** 2)  # (batch, 2, 112,112,80)

            consistency_weight = get_current_consistency_weight(iter_num // (80*60))

            threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            loss = 0.5 * (loss_seg + loss_seg_dice) + consistency_loss * consistency_weight

            print('loss:', loss)
            loss_train = loss_train + loss

            optimizer.zero_grad()
            loss.backward()  # student更新
            optimizer.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)  # 指数滑动平均给教师模型更新EMA平滑
            iter_num = iter_num + 1


            with torch.no_grad():
                outputs = model(volume_batch[:1])
                y_pred_dice = torch.argmax(outputs, dim=1)
                y_pred_dice = y_pred_dice.detach().cpu().numpy()
                label_batch1 = label_batch[:1].detach().cpu().numpy()

                dice1 = Dice(y_pred_dice, label_batch1)


                dice = dice + dice1

                print(dice)

        loss_train = loss_train
        loss_train = loss_train.item() / len(train_loader.dataset)

        dice_train = dice / len(train_loader.dataset)


        train_loss.append(loss_train)
        train_dice.append(dice_train)

        with open("E:/fei EZF CT/3DUnet/Semi_/loss.txt", 'w') as loss_train:
            loss_train.write(str(train_loss))

        with open("E:/fei EZF CT/3DUnet/Semi_/dice.txt", 'w') as dice_train:
            dice_train.write(str(train_dice))

        print('PPV', round(PPV, 4))
        print('sen', round(sen, 4))
        print('ASD',  round(ASD, 4))
        print('hd95', round(hd, 4))
        print('dice', round(dice, 4))
        model.eval()

        test_correct = 0
        test_total = 0
        test_running_loss = 0

        epoch_test_iou = []
        test_HD = []
        test_ASD = []
        test_ppv = []
        test_sen = []

        test_dice = []

        with torch.no_grad():
            for x, y in tqdm(testloader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = F.cross_entropy(y_pred, y)

                y_pred = torch.argmax(y_pred, dim=1)
                test_correct += (y_pred == y).sum().item()
                test_total += y.size(0)
                test_running_loss += loss.item()

                y_pred_dice = y_pred
                y_pred_dice = y_pred_dice.cpu().numpy()
                ys = y.cpu().numpy()

                dice = Dice(y_pred_dice, ys)
                print('testdice', dice)

                testhd = HD(y_pred_dice, ys)
                testASD = asd(y_pred_dice, ys)
                testPPV = ppv(y_pred_dice, ys)
                testsen = sensitivity(y_pred_dice, ys)

                test_dice.append(dice)
                test_HD.append(testhd)
                test_ASD.append(testASD)
                test_ppv.append(testPPV)
                test_sen.append(testsen)
                print('testhd',testhd)
                print('testASD', testASD)
                print('testPPV', testPPV)
                print('testsen', testsen)

        print(len(testloader.dataset))
        epoch_test_loss = test_running_loss / len(testloader.dataset)

        epoch_test_acc = test_correct / (test_total * 96 * 96 * 200)
        print('epoch_test_acc', epoch_test_acc)

