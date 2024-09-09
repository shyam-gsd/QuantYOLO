from ultralytics import YOLO
from ultralytics.data import YOLODataset
import torch
import matplotlib.pyplot as plt
import json
import pandas as pd
import yaml
from pathlib import Path
import numpy as np
import os
from sklearn.metrics import confusion_matrix



def calc_TP_FN(confMat, cls):
    tp = {}
    fn = {}
    fp = {}
    tn = {}

    print(confMat.shape)

    for c in cls:
        x = confMat[c,c]
        y = np.sum(confMat[:,c]) - x
        z = np.sum(confMat[c,:]) - x
        w = np.sum(confMat) - y - z - x 

        tp[c] = x
        fn[c] = y
        fp[c] = z
        tn[c] = w

    return tp,fn,fp,tn
if __name__ == "__main__":
    
       
    models_paths = [ "train74"] #"train74/weights/best.pt



    conf = yaml.safe_load(Path('data.yaml').read_text())
    dataset = YOLODataset(data=conf, img_path= "datasets/coco/images/val2017/")
    labels = dataset.get_labels()
    
    general_classes =  [0,2,3,7,9,10,42]

    map_vals = []
    tp_vals = []
    fn_vals = []

    for path in models_paths:
        dir_name = "val_plots/"+path+"/"
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        p = "runs/detect/"+path + "/weights/best.pt"
        model = YOLO(p)
        results = model.val(data="coco128.yaml",save_json= True,classes= general_classes,conf=0.25,imgsz=320)#
        # predictions_path = Path("/".join(results.save_dir.parts)+'/predictions.json')
        # with open(predictions_path) as f:
        #     predictions = json.load(f)



        
        cats = [results.names[c] for c in general_classes]
        y_pos = np.arange(len(general_classes))
        data = []
        for c in general_classes:
            num = 0
            for label in labels:
                num += list(label["cls"]).count(c)
            data.append(num)
        fig, ax = plt.subplots(figsize=(12,8))
        sorted_indices = np.argsort(data)
        sorted_categories = np.array(cats)[sorted_indices]
        sorted_values = np.array(data)[sorted_indices]
        ax.barh(sorted_categories,sorted_values,color="skyblue")
        for i in range(len(general_classes)):
            ax.text(sorted_values[i] + 5, y_pos[i], f'{sorted_values[i]}', va='center')
        ax.set_xlabel("Number of objects per class")
        ax.set_title("Ground Truth")
        ax.set_yticks(y_pos)
        plt.suptitle(f"{len(labels)} files and {len(general_classes)} classes")
        plt.tight_layout()
        plt.savefig(dir_name+"total_files.png")
        plt.clf()

        true_positives, false_negatives, false_positives, true_negatives = calc_TP_FN(results.confusion_matrix.matrix,general_classes)
        tp_values = [true_positives[cls] for cls in general_classes]
        fn_values = [false_negatives[cls] for cls in general_classes]
        fp_values = [false_positives[cls] for cls in general_classes]
        tn_values = [true_negatives[cls] for cls in general_classes]
        tp_vals.append(tp_values)
        fn_vals.append(fn_values)
        fig, ax = plt.subplots(figsize=(13,8))
        sorted_tp = np.array(tp_values)[sorted_indices]
        sorted_fn = np.array(fn_values)[sorted_indices]
        ax.barh(y_pos, sorted_fn, color='red', edgecolor='black', label='False Negative')
        ax.barh(y_pos, sorted_tp, left=sorted_fn, color='green', edgecolor='black', label='True Positive')
        for i in range(len(general_classes)):
            txt = ax.text(sorted_fn[i] + sorted_tp[i] + 5, y_pos[i] , f'{sorted_fn[i]}', va='top',color="red",fontweight="bold")
            ax.text(sorted_fn[i] + sorted_tp[i] + 5, y_pos[i] , f'{sorted_tp[i]}', va='bottom',color="green",fontweight="bold")
        ax.set_xlabel('Count')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_categories)
        ax.set_title('Detection results True Positives and False Negatives')
        plt.tight_layout()
        plt.savefig(dir_name+"TP_FN.png")
        plt.clf()
        

        p_values = [tp_values[c] / (tp_values[c] + fp_values[c]) for c in range(len(general_classes))]
        #p_values = [results.box.p[c] for c in range(len(general_classes))]
        fig, ax = plt.subplots(figsize=(13,8))
        sorted_p = np.array(p_values)[sorted_indices]
        ax.barh(sorted_categories,sorted_p,color="#cd853f")
        ax.set_xticks(np.linspace(0, 1, num=11))
        for i in range(len(general_classes)):
            ax.text(sorted_p[i] + 0.005, y_pos[i], f'{round(sorted_p[i],3)}',color="#cd853f", va='center',fontweight="bold")
        ax.set_title("Precision")
        ax.set_yticks(y_pos)
        plt.savefig(dir_name+"precision_per_class.png")
        plt.clf()

        r_values = [tp_values[c] / (tp_values[c] + fn_values[c]) for c in range(len(general_classes))]
        #r_values = [results.box.r[c] for c in range(len(general_classes))]
        fig, ax = plt.subplots(figsize=(13,8))
        sorted_r = np.array(r_values)[sorted_indices]
        ax.barh(sorted_categories,sorted_r,color="#cd853f")
        ax.set_xticks(np.linspace(0, 1, num=11))
        for i in range(len(general_classes)):
            ax.text(sorted_r[i] + 0.005, y_pos[i], f'{round(sorted_r[i],3)}',color="#cd853f", va='center',fontweight="bold")
        ax.set_title("recall")
        ax.set_yticks(y_pos)
        plt.savefig(dir_name+"recall_per_class.png")
        plt.clf()

        map_values = [results.box.ap50[c] for c in range(len(general_classes))]
        map_vals.append(map_values)
        fig, ax = plt.subplots(figsize=(13,8))
        sorted_map = np.array(map_values)[sorted_indices]
        ax.barh(sorted_categories,sorted_map,color="#4169e1")
        ax.set_xticks(np.linspace(0, 1, num=11))
        for i in range(len(general_classes)):
            ax.text(sorted_map[i] + 0.005, y_pos[i], f'{round(sorted_map[i],3)}',color="#4169e1", va='center',fontweight="bold")
        ax.set_title("mAP@50 per class")
        ax.set_yticks(y_pos)
        plt.savefig(dir_name+"map50_per_class.png")
        plt.clf()

        miss_rate = 1 - np.array(r_values)
        fppi = []
        fp = np.sum(results.confusion_matrix.matrix, axis=0) - np.diag(results.confusion_matrix.matrix)
        for d,c in zip(data,general_classes):
            fppi.append(fp[c] / d)
        log_fppi_values = np.logspace(-2, 0, num=9)
        interpolated_miss_rates = [np.interp(log_fppi_values, [f], [m]) for f, m in zip(fppi, miss_rate)]
        lamr = [np.exp(np.mean(np.log(imr))) for imr in interpolated_miss_rates]
        fig, ax = plt.subplots(figsize=(12,8))
        sorted_lamr = np.array(lamr)[sorted_indices]
        ax.barh(sorted_categories,sorted_lamr,color="#4169e1")
        for i in range(len(general_classes)):
            ax.text(sorted_lamr[i] + 0.005, y_pos[i], f'{round(sorted_lamr[i],3)}',color="#4169e1", va='center',fontweight="bold")
        
            #plt.scatter(fppi, miss_rate, marker='o')  # plot the original data points

        #plt.xscale('log')
        #plt.xlabel('False Positives Per Image (FPPI)')
        #plt.ylabel('Miss Rate')
        ax.set_title('Logarithmic Average Miss Rate')
        #plt.legend()
        #plt.grid(True)
        plt.savefig(dir_name+"lamr.png")
        plt.clf()

        for i,n in enumerate(general_classes):
            pc = results.box.p_curve[i]
            rc = results.box.r_curve[i]
            plt.plot(rc,pc,marker='.')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.fill_between(rc, pc, alpha=0.3)
            plt.xlim(0, max(rc))
            plt.ylim(0, max(pc))
            plt.xticks(np.linspace(0, 1, num=11))
            plt.title(f"class = {results.names[n]} Precision vs Recall mAP@50 = {round(results.box.ap50[i],3)}")
            plt.savefig(dir_name+results.names[n]+"_PR_curve.png")
            plt.clf()
    

    # fig, ax = plt.subplots(figsize=(13,8))
    # cats = [results.names[c] for c in general_classes]
    # bar_width = 0.4
    # x = np.arange(len(cats))
    # ax.barh(x,map_vals[0],height= bar_width,color="#4169e1",label="previous")
    # ax.barh(x+ bar_width,map_vals[1],height= bar_width,color="#ed7490",label="new model")
    
    # for i in range(len(general_classes)):
    #     ax.text(map_vals[0][i] + 0.005, x[i], f'{round(map_vals[0][i],3)}',color="#4169e1", va='center',fontweight="bold")
    #     ax.text(map_vals[1][i] + 0.005, x[i] + bar_width, f'{round(map_vals[1][i],3)}',color="#ed7490", va='center',fontweight="bold")
    # ax.set_title("mAP@50 per class")
    # ax.set_yticks(x + bar_width / 2, cats)
    # ax.set_yticklabels(cats)
    # ax.legend()
    # plt.savefig("map50_per_class_per_model.png")
    # plt.clf()

    # fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # # Plotting False Negatives
    # bars_fn_new = axs[0].barh(x - bar_width/2, fn_vals[1], height=bar_width, color="#e85656", label="False Negative New Model")
    # bars_fn_prev = axs[0].barh(x + bar_width/2, fn_vals[0], height=bar_width, color="#ed8e8e", label="False Negative Previous")
    
    # axs[0].set_title('False Negatives per Class')
    # axs[0].set_yticks(x)
    # axs[0].set_yticklabels(cats)
    # axs[0].set_xlabel('Count')
    # axs[0].legend(handles=[bars_fn_prev,bars_fn_new],labels=["False Negative Previous","False Negative New Model"])

    # # handles, labels = plt.gca().get_legend_handles_labels()
    # # print(handles)
    # # print(labels)
    # # order = [1,0]
    # # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    # for bar in bars_fn_prev:
    #     axs[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color="#ed8e8e", fontweight="bold")

    # for bar in bars_fn_new:
    #     axs[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color="#e85656", fontweight="bold")

    # # Plotting True Positives
    # bars_tp_new = axs[1].barh(x - bar_width/2, tp_vals[1], height=bar_width, color="#409c40", label="True Positive New Model")
    # bars_tp_prev = axs[1].barh(x + bar_width/2, tp_vals[0], height=bar_width, color="#73bd73", label="True Positive Previous")
    
    # axs[1].set_title('True Positives per Class')
    # axs[1].set_yticks(x)
    # axs[1].set_yticklabels(cats)
    # axs[1].set_xlabel('Count')
    # axs[1].legend(handles=[bars_tp_prev,bars_tp_new],labels=["True Positive Previous","True Positive New Model"])

    # for bar in bars_tp_prev:
    #     axs[1].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color="#73bd73", fontweight="bold")

    # for bar in bars_tp_new:
    #     axs[1].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color="#409c40", fontweight="bold")


    # plt.tight_layout()
    # plt.savefig("Both_models_TP_FN.png")
    # plt.clf()
        # print(results.curves)
        # print(results.results_dict)
        # print(results.class_result(0))
        # Assuming your JSON results are saved as 'results.json'
        
