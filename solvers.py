import network
from dataloader import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import lr_schedule
import utils
import torch.nn.functional as F
from modules import PseudoLabeledData, load_seed, load_seed_iv, split_data, z_score, normalize
import numpy as np
import adversarial
from utils import ConditionalEntropyLoss, LabelSmooth
from models import EMA
from cmd_1 import CMD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
import Adver_network
from typing import Optional, Tuple


def test_suda(loader, model):
    start_test = True
    with torch.no_grad():
        # 获得迭代数据
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            # 获得样本与标签
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            # 使用gpu
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels
            # 获得预测结果
            _, outputs = model(inputs)
            # 200个批次连接
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # 获得预测标签
    _, predictions = torch.max(all_output, 1)
    # 计算所有样本的acc
    accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

    y_true = all_label.cpu().data.numpy()
    y_pred = predictions.cpu().data.numpy()
    labels = np.unique(y_true)

    # 计算各种指标
    ytest = label_binarize(y_true, classes=labels)
    ypreds = label_binarize(y_pred, classes=labels)

    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
    matrix = confusion_matrix(y_true, y_pred)

    return accuracy, f1, auc, matrix


def test_muda(dataset_test, model):
    start_test = True
    features = None
    with torch.no_grad():

        for batch_idx, data in enumerate(dataset_test):
            Tx = data['Tx']
            Ty = data['Ty']
            Tx = Tx.float().cuda()

            # 获得预测结果
            feats, outputs = model(Tx)

            # 200个批次连接
            if start_test:
                all_output = outputs.float().cpu()
                all_label = Ty.float()
                features = feats.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, Ty.float()), 0)
                features = np.concatenate((features, feats.float().cpu()), 0)

            # 获得预测标签
        _, predictions = torch.max(all_output, 1)
        # 计算所有样本的acc
        accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

        y_true = all_label.cpu().data.numpy()
        y_pred = predictions.cpu().data.numpy()
        labels = np.unique(y_true)

        # 计算各种指标
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)

        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        matrix = confusion_matrix(y_true, y_pred)

        return accuracy, f1, auc, matrix, features, y_pred


def MFA_LR(args):
    """
    Parameters:
        @args: arguments
    """
    # --------------------------
    # 数据导入
    # --------------------------

    # 加载数据集SEED
    criterion = LabelSmooth(num_class=args.num_class).to(args.device)
    if args.dataset in ["seed", "seed-iv"]:
        print("DATA:", args.dataset, " SESSION:", args.session)
        if args.dataset == "seed":
            X, Y = load_seed(args.file_path, session=args.session, feature="de_LDS")
        else:
            # [1 session]
            if args.mixed_sessions == 'per_session':
                X, Y = load_seed_iv(args.file_path, session=args.session)
            # [3 sessions]
            elif args.mixed_sessions == 'mixed':
                X1, Y1 = load_seed_iv(args.file_path, session=1)
                X2, Y2 = load_seed_iv(args.file_path, session=2)
                X3, Y3 = load_seed_iv(args.file_path, session=3)

                X = {}
                Y = {}
                for key in X1.keys():
                    X1[key], _, _ = z_score(X1[key])
                    X2[key], _, _ = z_score(X2[key])
                    X3[key], _, _ = z_score(X3[key])

                    X[key] = np.concatenate((X1[key], X2[key], X3[key]), axis=0)
                    Y[key] = np.concatenate((Y1[key], Y2[key], Y3[key]), axis=0)
            else:
                print("Option [mixed_sessions] is not valid.")
                exit(-1)

        # 挑选出目标域
        trg_subj = args.target - 1
        #目标域数据
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # subjects
        subject_ids = X.keys()
        num_domains = len(subject_ids)

        # [Option 1]: Evaluation over all target domain
        Vx = Tx
        Vy = Ty

        # [Option 2]: Evaluation over test data from Target domain
        # Split target data for testing
        # Tx, Ty, Vx, Vy = split_data(Tx, Ty, args.seed, test_size=0.2)
        # Standardize target data
        Tx, m, std = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=std)


        print("Target subject:", trg_subj)
        print("Tx:", Tx.shape, " Ty:", Ty.shape)
        print("Vx:", Vx.shape, " Vy:", Vy.shape)
        print("Num. domains:", num_domains)

        print("Data were succesfully loaded")

        # Train dataset
        train_loader = UnalignedDataLoader()
        train_loader.initialize(num_domains, X, Y, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
        datasets = train_loader.load_data()

        #classes = np.unique(Ty)

        # Test dataset
        test_loader = UnalignedDataLoaderTesting()
        test_loader.initialize(Vx, Vy, 200, shuffle_testing=False, drop_last_testing=False)
        dataset_test = test_loader.load_data()

    else:
        print("This dataset does not exist.")
        exit(-1)


    # --------------------------
    # Create Deep Neural Network
    # --------------------------
    # For synthetic dataset
    if args.dataset in ["seed", "seed-iv"]:
        # Define Neural Network
        # 2790 for SEED
        # 620 for SEED-IV
        input_size = 3720 if args.dataset == "seed" else 620   # windows_size=9
        # hidden_size = 310

        model = network.DFN(input_size=input_size, hidden_size=args.hidden_size, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class, radius=args.radius).cuda()
        decoders = [
                    network.Decoder(hidden_size=args.bottleneck_dim, out_dim=3720).to(args.device)
                      for j in range(num_domains - 1)
                    ]
        adv_net = network.DiscriminatorDANN(in_feature=model.output_num(), radius=10.0, hidden_size=args.bottleneck_dim, max_iter=1000).cuda()
    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)

    #
    parameter_classifier = [model.get_parameters()[2]]
    # parameter_feature = model.get_parameters()[0:2]
    parameter_feature = model.get_parameters()[0:2] + adv_net.get_parameters()# + decoder.get_parameters() for decoder in decoders
    for k in range(num_domains - 1):
        parameter_feature += decoders[k].get_parameters()     
    optimizer_classifier = torch.optim.SGD(parameter_classifier, lr=args.lr_a, momentum=0.9, weight_decay=0.005)
    optimizer_feature = torch.optim.SGD(parameter_feature, lr=args.lr_a, momentum=0.9, weight_decay=0.005)

    # if gpus are availables
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    # ------------------------
    # Model training
    # ------------------------

    # Number of centroids for semantic loss
    if args.dataset in ["seed", "seed-iv"]:
        Cs_memory = []
        for d in range(num_domains):
            Cs_memory.append(torch.zeros(args.num_class, args.bottleneck_dim).cuda())
        Ct_memory = torch.zeros(args.num_class, args.bottleneck_dim).cuda()

    else:
        print("SETTING number of centroids: The dataset does not exist.")
        exit()

    cent = ConditionalEntropyLoss().cuda()

    ''' Exponential moving average (simulating teacher model) '''
    ema = EMA(0.998)
    ema.register(model)

    # for weighting loss
    weights_d = torch.zeros(num_domains - 1).cuda()
    weights_d += 1

    weights_s = torch.zeros(num_domains - 1).cuda()
    weights_s += 1

    alpha = 0.90

    # [CMD]
    cmd = CMD(n_moments=2)
    log_total_loss = []
    my_grl = adversarial.AdversarialLayer()
    my_recon = utils.CosineSimilarityLoss().to(args.device)
    for i in range(args.max_iter1):

        for batch_idx, data in enumerate(datasets):
            # get the source batches
            x_src = list()
            y_src = list()
            d_src = list()
            index = 0

            for domain_idx in range(num_domains - 1):
                tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
                tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
                # labels = torch.from_numpy(np.array([[index] * args.batch_size]).T).type(torch.FloatTensor).flatten().long().cuda()
                x_src.append(tmp_x)
                # d_src.append(labels)
                y_src.append(tmp_y)

            # get the target batch
            x_trg = data['Tx'].float().cuda()
            # print(x_trg.shape)
            # Enable model to train
            model.train(True)
            adv_net.train(True)

            # obtain schedule for learning rate
            optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier, i, lr=args.lr_a)
            optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i, lr=args.lr_a)

            # Get features target
            features_target, outputs_target = model(x_trg)
            # pseudo-labels
            pseu_labels_target = torch.argmax(outputs_target, dim=1)
            
            rec_loss = 0
            my_recon_loss = 0
            mixSubjectFeature = []
            mixMasks = []
            batch_size, timeWin, num_channels, pindai = x_src[0].shape
            # print(batch_size, timeWin, num_channels, pindai)
            # x_src = x_src.view(batch_size, num_channels, timeWin, width)
            for k in range(num_domains - 1):
                masks = utils.generate_channel_masks(x_src[k].view(batch_size, num_channels, timeWin, pindai), mask_ratio=0.50)
                # print(masks.shape,x_src[k].shape)
                mid_feat, _ = model(x_src[k]*masks)
                rec_loss += utils.marginal(mid_feat, features_target)
                x_out = decoders[k](mid_feat)
                x_out = x_out.view(x_src[k].shape)
                mixSubjectFeature.append(x_out.cuda())
                mixMasks.append(masks.cuda())
                my_recon_loss += my_recon((x_out * (1 - masks)).squeeze(),x_trg.view(x_trg.size(0), -1))

            # for m in range(num_domains - 1):
            #     shared_last_out_2, _ = model(mixSubjectFeature[m])
            #     # x_out = decoders[k](shared_last_out_2)
            #     print((mixSubjectFeature[m] * (1 - mixMasks[m])).squeeze().shape)
            #     rec_loss += utils.marginal((mixSubjectFeature[m] * (1 - mixMasks[m])).view(mixSubjectFeature[m].size(0), -1),x_trg.view(x_trg.size(0), -1))
            rec_loss /= num_domains - 1
            my_recon_loss /= num_domains - 1

            sm_loss = []
            dom_loss = []
            pred_src = []
            pred_src_domain = []
            feats = []
            for domain_idx in range(num_domains - 1):
                features_source, outputs_source = model(x_src[domain_idx])
                pred_src.append(outputs_source)
                feats.append(features_source)

            # Stack/Concat data from each source domain
            feature_outs = torch.cat(feats, dim=0)
            all_features = torch.cat((feature_outs, features_target), dim=0)
            pred_source = torch.cat(pred_src, dim=0)
            labels_source = torch.cat(y_src, dim=0)
            adv_loss = utils.loss_adv(my_grl.apply(all_features), adv_net, logits=torch.nn.Softmax(dim=1)(pred_source).detach())

            # [COARSE-grained training loss]
            classifier_loss = criterion(pred_source, labels_source.flatten())

            # [1] total_loss = classifier_loss + align_loss + 0.1 * loss_trg_cent
            total_loss = classifier_loss + 0.5 * rec_loss + 0.5*adv_loss + 0.5* my_recon_loss

            # Reset gradients
            optimizer_classifier.zero_grad()
            optimizer_feature.zero_grad()

            total_loss.backward()

            optimizer_classifier.step()
            optimizer_feature.step()


            # free variables
            for d in range(num_domains):
                Cs_memory[d].detach_()
            Ct_memory.detach_()

        # set model to test
        model.train(False)

        # calculate accuracy performance
        best_acc, best_f1, best_auc, best_mat, features, labels = test_muda(dataset_test, model)
        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}\t loss: {:.4f}".format(i, best_acc, best_f1, best_auc, total_loss)
        args.log_file.write(log_str)
        args.log_file.flush()
        print(log_str)
        log_total_loss.append(total_loss.data)

    return X, Y, best_acc, best_f1, best_auc, best_mat, model, log_total_loss



def RSDA(X, Y, model, args):

    # prepare data
    dset_loaders = {}

    if args.dataset in ["seed", "seed-iv"]:

        print("DATA:", args.dataset, " SESSION:", args.session)

        # get dictionary keys
        subjects = X.keys()

        print(subjects)

        # build Source dataset
        Sx = Sy = None
        i = 0
        flag = False
        selected_subject = args.target - 1
        trg_subj = -1

        for s in subjects:
            # if subject is not the selected for target
            if i != selected_subject:

                tr_x = np.array(X[s])
                tr_y = np.array(Y[s])

                # global-wise standardization
                tr_x, m, std = z_score(tr_x)

                if not flag:
                    Sx = tr_x
                    Sy = tr_y
                    flag = True
                else:
                    Sx = np.concatenate((Sx, tr_x), axis=0)
                    Sy = np.concatenate((Sy, tr_y), axis=0)
            else:
                # store ID
                trg_subj = s
            i += 1

        print("[+] Target subject:", trg_subj)

        # Target dataset
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])
        Vx = Tx
        Vy = Ty
        # Split target data for testing
        # Tx, Ty, Vx, Vy = split_data(Tx, Ty, args.seed, test_size=0.2)

        # Global-wise standardization
        Tx, m, sd = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=sd)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Vx.shape, "Ty_test:", Vy.shape)

        # to tensor
        Sx_tensor = torch.tensor(Sx)
        Sy_tensor = torch.tensor(Sy)

        # create containers for source data
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)

        # create container for target data
        # target_tr = PseudoLabeledData(samples.numpy(), weighted_pseu_label, weights)

        # create container for test data
        Vx_tensor = torch.tensor(Vx)
        Vy_tensor = torch.tensor(Vy)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

        # data loader
        dset_loaders["source"] = DataLoader(source_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["target"] = DataLoader(target_ts, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["test"] = DataLoader(target_ts, batch_size=200, shuffle=False, num_workers=4)

        print("Data were succesfully loaded")

    else:
        print("This dataset does not exist.")
        exit()
    criterion = LabelSmooth(num_class=args.num_class).to(args.device)

    # Create model
    if args.dataset in ["seed", "seed-iv"]:

        # [Window]
        input_size = 3720 if args.dataset == "seed" else 620   # windows_size=9

        # model = network.DFN(input_size=input_size, hidden_size=args.hidden_size, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class, radius=10.0).cuda()

        # setting Adversarial net
        # adv_net = network.DiscriminatorDANN(in_feature=model.output_num(), radius=10.0, hidden_size=args.bottleneck_dim, max_iter=1000).cuda()

    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)

    # Ger trainable weights
    parameter_classifier = [model.get_parameters()[2]]
    parameter_feature = model.get_parameters()[0:2] #+ adv_net.get_parameters()

    # gradient reversal layer
    my_grl = adversarial.AdversarialLayer()

    optimizer_classifier = torch.optim.Adam(parameter_classifier, lr=args.lr_b, weight_decay=0.005)
    optimizer_feature = torch.optim.Adam(parameter_feature, lr=args.lr_b, weight_decay=0.005)

    # if number of GPUS is greater 1
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        # adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## Train MODEL

    # lenght of data
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])

    # auxiliar variables
    best_acc = 0.0

    # centroids for each cluster
    if args.dataset in ["seed", "seed-iv"]:
        Cs_memory = torch.zeros(args.num_class, args.bottleneck_dim).cuda()
        Ct_memory = torch.zeros(args.num_class, args.bottleneck_dim).cuda()

    else:
        print("The number of centroids for this dataset has not been selected yet.")
        exit()

    ''' Exponential moving average (simulating teacher model) '''
    ema = EMA(0.998)
    ema.register(model)
    final_acc = 0
    final_f1 = 0
    final_auc = 0
    final_mat = []

    # iterate over
    for i in range(args.max_iter2):

        # Testing phase
        if i % 1 == 0:
            # set model training to False
            model.train(False)
            # calculate accuracy on test set
            best_acc, best_f1, best_auc, best_mat = test_suda(dset_loaders, model)
            if final_acc < best_acc:
                final_acc = best_acc
                final_f1 = best_f1
                final_auc = best_auc
                final_mat = best_mat
                
            if i == 0:
                log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(i, best_acc, best_f1, best_auc)
            else: 
                log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f} \t loss: {:.4f}".format(i, best_acc, best_f1, best_auc, total_loss.item())
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)

        # Enable model for training
        model.train(True)
        # adv_net.train(True)

        # obtain schedule for learning rate
        optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier, i, lr=args.lr_b)
        optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i, lr=args.lr_b)

        # get data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        # Get batch for source and target domains
        inputs_source_, labels_source = next(iter_source)
        inputs_target_, ture_labels_target = next(iter_target)
        # Cast
        inputs_source_ = inputs_source_.type(torch.FloatTensor)
        labels_source = labels_source.type(torch.LongTensor)
        inputs_target_ = inputs_target_.type(torch.FloatTensor)
        ture_labels_target = ture_labels_target.type(torch.LongTensor)
        # to cuda
        inputs_source, labels_source = inputs_source_.cuda(), labels_source.cuda()
        inputs_target, ture_labels_target = inputs_target_.cuda(), ture_labels_target.cuda()
        # weights = weights.type(torch.Tensor).cuda()

        # weights[weights < 0.5] = 0.0

        # get features and labels for source and target domain
        features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)

        # concatenate features
        features = torch.cat((features_source, features_target), dim=0)
        # concatenate logits
        logits = torch.cat((outputs_source, outputs_target), dim=0)

        # cross-entropy loss
        source_class_loss = criterion(outputs_source, labels_source.flatten())

        # adversarial loss
        # adv_loss = utils.loss_adv(my_grl.apply(features), adv_net, logits=torch.nn.Softmax(dim=1)(logits).detach())
        # obtain pseudo labels
        # pseu_labels_target = torch.argmax(outputs_target, dim=1)
        # [Conditional entropy]
        ce_loss = torch.mean(utils.Entropy(F.softmax(outputs_target, dim=1)))

        # function robust loss
        # target_robust_loss = utils.robust_pseudo_loss(outputs_target, pseu_labels_target)

        # classifier loss
        classifier_loss = source_class_loss #+ target_robust_loss


        # semantic loss
        # loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target, Cs_memory, Ct_memory, decay=0.9)
        mmd_t_loss = utils.conditional(
                   features_source,
                   features_target,
                   labels_source.reshape((args.batch_size, 1)),
                   torch.nn.functional.softmax(outputs_target,dim = 1),
                   2.0,
                   5,
                   None)
        # [FINAL LOSS]
        # [original]
        #total_loss = classifier_loss + 0.1 * adv_loss + 0.1 * loss_sm + 0.1 * ce_loss
        # [set]
        total_loss = classifier_loss + 0.5 * mmd_t_loss + 0.1 * ce_loss

        # reset gradients
        optimizer_classifier.zero_grad()
        optimizer_feature.zero_grad()

        # compute gradients
        total_loss.backward()

        # update weights
        optimizer_feature.step()
        optimizer_classifier.step()

        # Polyak averaging.
        ema(model)  # TODO: move ema into the optimizer step fn.

        Cs_memory.detach_()
        Ct_memory.detach_()

    return final_acc, final_f1, final_auc, final_mat, model



