import sys
sys.path.append('/clusterhome/clusteruser11/QuantYOLO/ultralytics/ultralytics')
sys.path.append('/clusterhome/clusteruser11/QuantYOLO/brevitas/src')
sys.path.append('/clusterhome/clusteruser11/QuantYOLO/brevitas/src/brevitas')
sys.path.append('/clusterhome/clusteruser11/QuantYOLO/brevitas')
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG,LOGGER,colorstr
import numpy as np
import random 
from pathlib import Path
import time
import threading
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os
import json
import math
import torch
import traceback
from torchvision import transforms, datasets
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode
import gc

'''
Function to Check if the array of numbers has a plateau

Args:
    arr : array of the numbers
    window_size : total number of elements that are to be averaged 
    tolerance : if the smoothened curve remains in this range for 3 consiquetive times then it is a plateau
'''
def has_plateau(arr, window_size=3, tolerance=0.1):
    if len(arr) < 3:  # If the array has fewer than 3 elements, it can't have a plateau
        return False

    # Calculate the moving average
    smoothed_arr = gaussian_filter1d(arr, sigma=6)
    
    counter = 1  # Initialize a counter for the current sequence length
    for i in range(1, len(smoothed_arr)):
        if abs(smoothed_arr[i] - smoothed_arr[i - 1]) <= tolerance:  # Check if difference is within the tolerance
            counter += 1  # Increment counter if they are within tolerance
            if counter >= 3:  # Check if the sequence length is at least 3
                return True
        else:
            counter = 1  # Reset the counter if the sequence breaks
    
    return False  # Return False if no plateau is found


class ModelOnPlateau(Exception):
    pass

class Tuner():
    
    '''
        __init__ and _mutate fucntions are directly taken from ultralytics
    '''

    def __init__(self,meta,args=DEFAULT_CFG,**kwargs):
        """
        Initialize the Tuner with configurations.

        Args:
            args (dict, optional): Configuration for hyperparameter evolution.
        """
        self.space = {  # key: (min, max, gain(optional))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            "box": (1.0, 20.0),  # box loss gain
            "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
            "dfl": (0.4, 6.0),  # dfl loss gain
            "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (0.0, 45.0),  # image rotation (+/- deg)
            "translate": (0.0, 0.9),  # image translation (+/- fraction)
            "scale": (0.0, 0.95),  # image scale (+/- gain)
            "shear": (0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (0.05, 1.0),  # image flip up-down (probability)
            "fliplr": (0.05, 1.0),  # image flip left-right (probability)
            "bgr": (0.05, 1.0),  # image channel bgr (probability)
            "mosaic": (0.05, 1.0),  # image mixup (probability)
            "mixup": (0.05, 1.0),  # image mixup (probability)
            "copy_paste": (0.05, 1.0),  # segment copy-paste (probability)
            "batch" : (16,128,16)
        }
        self.args = get_cfg(overrides=kwargs)
        self.tune_csv = Path("tune.csv") # file to store hyperparameters data, it stores all the hyps and the best fitness value recieved till the model went to plateau during training
        self.hyp_file = Path("hyp.yaml") # file to store hyperparameters
        self.prefix = colorstr("Tuner: ")
        
        # Custom Parameters
        
        self.patience = 5 # patience : number of epochs to wait before updating hyperparameters
        self.patience_tolerence = 0.001 # tolerence : if the target metric remains in this range it will be considered plateau
        self.patience_window_size = 3 # window_siye : total number of elements to be averaged 

        
        self.hyp = dict() # current hyperparameters

        self.process_id = None # current train thread
        self.model = None # currrent running model
        self.data = '' # data to be trained on 
        
        
        self.metrics_per_epoch = pd.DataFrame() # dataframe that keeps the record of metrics after each training epoch
        self.best_metric = dict() # holds best metrics

        '''
         number of epochs to wait before checking for plateau, 
         I used this in case the model does not directly start showing improvement when the training starts
         As we are updating hyperparameters in between
        '''
        self.cool_period = 10 
        
        

        self.exp_dir = "./experiments"
        self.target_metric = 'fitness'
        
        self.pat_cnt = 5 # counter to keep track of patience number
        self.epoch_cnt = 0 # total number of epochs the model trained ignoring the change in hyperparameters
        self.current_exp_epoch_cnt = 0 # counter to keep track of epochs after changing hyperparameters
        self.hyp_update_count = 0
        self.is_current_hyp_stale = False # this is bool to stop the current train process and update the hyperparameters
        self.best_model_path = None
        

        '''
        Following code creates necessary files to manage and save results  and metadata       
        '''
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.exp_run = 0
        
        for dir in os.listdir(self.exp_dir):
            name,exp = dir.split("_")
            self.exp_run += 1
        
        self.exp_path = self.exp_dir+"/exp_"+str(self.exp_run)+"/"
        print("experiment path.. "+self.exp_path)
        os.makedirs(self.exp_path)

        self.thread = None

        self.meta = meta

        
        


    def update_best_model_path(self):
        dir_runs = Path("runs/detect/")
        fols = sorted([f for f in dir_runs.iterdir() if f.is_dir() and f.name.startswith("train")])

        best_map = 0.0
        for f in fols:
            try:
                csv = pd.read_csv(f / "results.csv", sep=",")
                map = csv["    metrics/mAP50-95(B)"].max()
                if map > best_map:                    
                    if Path(f / "weights/best.pt").exists():
                        best_map = map
                        self.best_model_path = f / "weights/best.pt"
            except:
                pass
        print("updated path to "+str(self.best_model_path))
        
    
    def _mutate(self, parent="single", n=5, mutation=0.8, sigma=0.2):
        """
        Mutates the hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
            parent (str): Parent selection method: 'single' or 'weighted'.
            n (int): Number of parents to consider.
            mutation (float): Probability of a parameter mutation in any given iteration.
            sigma (float): Standard deviation for Gaussian random number generator.

        Returns:
            (dict): A dictionary containing mutated hyperparameters.
        """
        if self.tune_csv.exists():  # if CSV file exists: select best hyps and mutate
            # Select parent(s)
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]  # first column
            n = min(n, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness)][:n]  # top n mutations
            w = x[:, 0] - x[:, 0].min() + 1e-6  # weights (sum > 0)
            if parent == "single" or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == "weighted":
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            r = np.random  # method
            r.seed(int(time.time()))
            g = np.array([v[2] if len(v) == 3 else 1.0 for k, v in self.space.items()])  # gains 0-1
            ng = len(self.space)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (r.random(ng) < mutation) * r.randn(ng) * r.random() * sigma + 1).clip(0.3, 3.0)
            hyp = {k: float(x[i + 1] * v[i]) for i, k in enumerate(self.space.keys())}
        else:
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}

        # Constrain to limits
        for k, v in self.space.items():
            hyp[k] = max(hyp[k], v[0])  # lower limit
            hyp[k] = min(hyp[k], v[1])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

        # fix batch size make it int
        hyp["batch"] = 16 #int(hyp["batch"])


        self.hyp = hyp

        with open(self.hyp_file, "w") as f:
            f.write(str(hyp))

        self.hyp_update_count += 1
        return hyp


    

    '''
    initializes for first time the parameters
    this function  runs only 1 time before starting training
    '''
    def InitTrain(self,model,data,epochs,patience,imgsz,optim="SGD"):
        
        self.patience = patience
        self.pat_cnt = patience
        self.model = model
        self.data = data
        self.epochs = epochs
        self.imgsz = imgsz
        self.model.add_callback("on_train_epoch_end",self.onTrainEpochComplete)
        self.model.add_callback("on_model_save",self.onModelSaved)
        self.optim = optim
        
        
        self.epoch_cnt = 0
        
        hyp = self._mutate()
        self.meta.data = data
        self.meta.model = model.model_name
        self.meta.epochs = epochs
        self.meta.patience = patience
        self.meta.imgsz = imgsz
        self.meta.optimizer = optim
        self.meta.cool_period = self.cool_period
        self.meta.target_metrics = self.target_metric
        self.meta.best_model = self.best_model_path
        self.meta.highest_train_epochs = self.epoch_cnt
        
        
    
    '''
    this function starts the train loop,
    It spawns a thread and also an event to stop the training when required
    '''
    def StartTrain(self):        
        print("starting ")
        self.current_exp_epoch_cnt = 0
        del self.thread
        self.thread = threading.Thread(target=self.trainTask)
        
        self.thread.start()
        print("started")
        self.thread.join()

        if(self.current_exp_epoch_cnt < self.epochs):
            mutated_hyp = self._mutate()
            
            #self.thread = None

            self.model = None
            gc.collect()  # Collect garbage
            if self.meta.project == "FloatingPointTuning":
                self.update_best_model_path()
            else:
                self.best_model_path = "newyolo_def.yaml"
            self.model = YOLO(self.best_model_path)
            self.model.add_callback("on_train_epoch_end",self.onTrainEpochComplete)
            self.model.add_callback("on_model_save",self.onModelSaved)
            self.is_current_hyp_stale = False
            self.StartTrain()
        #model.train(data=data,epochs=epochs, plots=True, save=True, cfg=self.hyp_file,val=True, resume=True)


    '''
    Main train task, this function is run inside thread. it starts the training process
    '''
    def trainTask(self):
        try:
            self.model.train(data = self.data,epochs= self.epochs,lr0=0.00001,save= True,device=[0],cfg=self.hyp_file,val= True, imgsz= self.imgsz, optimizer=self.optim,project=self.meta.project)
        except KeyboardInterrupt:
            self.plot_res()
        except Exception as e:
            print("Training was cancelled. "+str(e))
            traceback.print_exc()
            


    

    def onModelSaved(self,trainer):
        # updates hyperparameters and starts training loop
        if(self.is_current_hyp_stale):
            raise ModelOnPlateau("Model reached a plateau")
            
            


    '''
    This function is called after each training epoch is completed
    If the model reaches plateau it stops the thread, updates the hyperparameters and then starts training process again. 
    '''
    def onTrainEpochComplete(self,trainer):
        #increment necessary counters
        self.epoch_cnt += 1
        self.current_exp_epoch_cnt += 1
        
        self.meta.highest_train_epochs = self.epoch_cnt
        # I skipped the first iteration as it was producing unecessary spike in the plot, because all metrics are 0 at the start of training. 
        if(self.current_exp_epoch_cnt <= 1):
            return
        

        # fetch metrics from trainer and save best metrics 
        metrics_data = trainer.metrics
        metrics_data["fitness"] = trainer.fitness if trainer.fitness else 0
        
        self.metrics_per_epoch = pd.concat([self.metrics_per_epoch, pd.DataFrame([metrics_data])],ignore_index=True)
        

        target_metric_list = self.metrics_per_epoch[self.target_metric].to_list()

        if(metrics_data[self.target_metric] >= max(target_metric_list)):
            #self.model.save(self.exp_path+"best.pt")
            id = self.metrics_per_epoch[self.target_metric].idxmax()
            self.best_metric = self.metrics_per_epoch.iloc[id].to_dict()
        

        # waits for cooling period and then checks for plateau

        mutated_hyp = self.hyp
        if(self.current_exp_epoch_cnt > self.cool_period):
            if(has_plateau(target_metric_list[:-(self.patience +self.cool_period)],self.patience_window_size,self.patience_tolerence)): # check only for the last 10 or self.cooling_period + self.patience of data to check if it has plateau
                self.pat_cnt -= 1
                if(self.pat_cnt <= 0):

                    # saves the hyperparameters and best metrics to tune.csv
                    best_fitness = self.best_metric["fitness"]
                    
                    log_row = [round(best_fitness, 5)] + [mutated_hyp[k] for k in self.space.keys()] + list(self.best_metric.values())
                    headers = "" if self.tune_csv.exists() else (",".join(["fitness"] + list(self.space.keys()) + list(self.best_metric.keys())) + "\n")
                    with open(self.tune_csv, "a") as f:
                        f.write(headers + ",".join(map(str, log_row)) + "\n")
                    self.pat_cnt = self.patience
                    self.plot_res()

                    # sets the bool for the to stop current train loop
                    self.best_model_path = trainer.best if trainer.best else trainer.last
                    self.is_current_hyp_stale = True
                    
            else:
                self.pat_cnt = self.patience
                    

        # for k in trainer.args:
        #     if(k[0] == 'lr0'):
        #         LOGGER.info(k)

        


    '''
    plots the training progress to graph
    '''
    def plot_res(self):
        df = pd.DataFrame(self.metrics_per_epoch)
        num_columns = df.shape[1]
        # Determine the number of rows needed for an Lx4 grid
        L = math.ceil(num_columns / 4)
        
        fig, axes = plt.subplots(L, 4, figsize=(20, 5 * L))
        
        # Flatten axes for easy iteration
        axes = axes.flatten()
        
        x = np.arange(df.shape[0])
        
        for idx, column in enumerate(df.columns):
            axes[idx].plot(x, df[column], label=f'{column} data')
            smooth_data = gaussian_filter1d(df[column], sigma=2)
            axes[idx].plot(x, smooth_data, label=f'{column} smoothed', linestyle='--')
            
            axes[idx].set_title(f'{column}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(column)
            axes[idx].legend()
        
        # Hide any unused subplots
        for i in range(num_columns, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.exp_path+"exp_plot_"+str(self.exp_run)+'_'+str(self.epoch_cnt)+".png")
        
        with open(self.exp_path+"meta.json", "w") as json_file:
            json.dump(self.meta, json_file, indent=4)
        
    
    
        

class Meta:
    def __init__(self, data, model, epochs, patience, cool_period, target_metrics, best_model, highest_train_epochs, imgsz, optimizer,project):
        self.data = data
        self.model = model
        self.epochs = epochs
        self.patience = patience
        self.cool_period = cool_period
        self.target_metrics = target_metrics
        self.best_model = best_model
        self.highest_train_epochs = highest_train_epochs
        self.imgsz = imgsz
        self.optimizer = optimizer
        self.project = project
def get_dataloader(folder, size):
    # dataset = datasets.ImageFolder(folder, transforms.ToTensor())
    dataset = datasets.ImageFolder(folder, transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ]))
    return torch.utils.data.DataLoader(dataset)
if __name__ == '__main__':
    # meta data to be updated as per experiment
    meta = {"data":"coco.yaml", "model":"runs/detect/train74/weights/best.pt", "epochs":1000,"patience":10,"cool_period":10, "target_metrics":"fitness","best_model":"train46", "highest_train_epochs" : 1000, "imgsz":320,"optimizer":"SGD","project":"Quantization"}

    # defining model
    
    meta = Meta(**meta)
    #model.load("runs/detect/train74/weights/best.pt")

    #instantiation of Tuner
    tuner = Tuner(meta)
    tuner.update_best_model_path()

    floatmodel = YOLO(tuner.best_model_path, "detect") #../repo/runs/detect/train46/weights/best.pt

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")



    quantModel = YOLO("newyolo_def.yaml")
    quantModel.load(tuner.best_model_path)


    print(quantModel.info())

    # dataloader = get_dataloader("ultralytics/images", 320)
    # with torch.no_grad():
    #     print("Calibrate:")
    #     with calibration_mode(quantModel):
    #         for x, _ in tqdm(dataloader):
    #             x = quantModel(x.to(device))
    #
    #     print("Bias Correction:")
    #     with bias_correction_mode(quantModel), bias_correction_mode(quantModel):
    #         for x, _ in tqdm(dataloader):
    #             x = quantModel(x.to(device))


    tuner.InitTrain(quantModel,data='coco128.yaml',epochs=1000,patience=10,imgsz=320)

    #start training
    tuner.StartTrain()

    #plot final results
    tuner.plot_res()


#sbatch
