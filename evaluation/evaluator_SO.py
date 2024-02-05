import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from evaluation.labels import labels
"""
Evaluator for interactive single-object segmentation
"""
class EvaluatorSO():

    def __init__(
        self,
        dataset,
        object_list_file,
        object_classes_list_file,
        result_file,
        MAX_IOU):
        
        self.dataset = dataset
        self.MAX_IOU = MAX_IOU
        self.label_all = labels[dataset]
        self.dataset_list = np.load(object_list_file)
        self.dataset_classes = np.loadtxt(object_classes_list_file, dtype=str)
        self.result_file = result_file

    def eval_per_class(self, label=None, MAX_IOU=0.8, dataset_=None, dataset_classes=None, exclude_classes=['wall','ceiling','floor','unlabelled','unlabeled']):
        objects = {}

        if exclude_classes:
            print('total number of objects: ', np.shape(dataset_classes))
            mask = np.isin(dataset_classes, exclude_classes , invert=True)
            print('number of objects kept: ',sum(mask))
            dataset_ = dataset_[mask]
            dataset_classes = dataset_classes[mask]
            for ii in dataset_:
                objects[ii[0].replace('scene','')+'_'+ii[1]]=1
        else:

            for ii in dataset_:
                objects[ii[0].replace('scene','')+'_'+ii[1]]=1
            print('number of objects kept: ',len(objects))


        if label:
            dataset_ = dataset_[dataset_classes == label]
            objects = {}
            for ii in dataset_:
                objects[ii[0].replace('scene','')+'_'+ii[1]]=1

        results_dict_KatIOU = {}
        num_objects = 0
        ordered_clicks = []

        all_object={}
        results_dict_per_click = {}
        results_dict_per_click_iou = {}
        all={}
        with open(self.result_file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                splits = line.rstrip().split(' ')
                scene_name = splits[1].replace('scene','')
                object_id = splits[2]
                num_clicks = splits[3]
                iou=splits[4]

                if (scene_name + '_' + object_id) in objects:
                    if (scene_name + '_' + object_id) not in all_object:
                        all_object[(scene_name + '_' + object_id)]=1
                        all[(scene_name + '_' + object_id)]=[]
                    all[(scene_name + '_' + object_id)].append((num_clicks,iou))


                    if float(iou)>=MAX_IOU:
                        if (scene_name+'_'+object_id) not in results_dict_KatIOU:
                            results_dict_KatIOU[scene_name+'_'+object_id]=float(num_clicks)
                            num_objects+=1
                            ordered_clicks.append(float(num_clicks))

                    elif int(num_clicks)>=20 and (float(iou)>=0):
                        if (scene_name+'_'+object_id) not in results_dict_KatIOU:
                            results_dict_KatIOU[scene_name+'_'+object_id] = float(num_clicks)
                            num_objects += 1
                            ordered_clicks.append(float(num_clicks))

                    results_dict_per_click.setdefault(num_clicks, 0)
                    results_dict_per_click_iou.setdefault(num_clicks, 0)

                    results_dict_per_click[num_clicks]+=1
                    results_dict_per_click_iou[num_clicks]+=float(iou)
                else:
                    #print(scene_name + '_' + object_id)
                    pass
        if len(results_dict_KatIOU.values())==0:
            print('no objects to eval')
            return 0


        click_at_IoU =sum(results_dict_KatIOU.values())/len(results_dict_KatIOU.values())
        print('click@', MAX_IOU, click_at_IoU, num_objects, len(results_dict_KatIOU.values()))


        return ordered_clicks, sum(results_dict_KatIOU.values()), len(results_dict_KatIOU.values()), results_dict_per_click_iou, results_dict_per_click 


    def eval_results(self):
        print('--------- Evaluating -----------')
        NOC = {}
        NOO = {}
      
        for iou_max in self.MAX_IOU:
            NOC[iou_max] = []
            NOO[iou_max] = []
            IOU_PER_CLICK_dict = None
            NOO_PER_CLICK_dict = None

            for l in list(set(self.label_all)):    
            
                _, noc_perclass, noo_perclass, iou_per_click, noo_per_click = self.eval_per_class(l, iou_max, self.dataset_list, self.dataset_classes, exclude_classes=None)
                NOC[iou_max].append(noc_perclass)
                NOO[iou_max].append(noo_perclass)

                if IOU_PER_CLICK_dict == None:
                    IOU_PER_CLICK_dict = iou_per_click
                else:
                    for k in IOU_PER_CLICK_dict.keys():
                        IOU_PER_CLICK_dict[k] += iou_per_click[k]

                if NOO_PER_CLICK_dict == None:
                    NOO_PER_CLICK_dict = noo_per_click
                else:
                    for k in NOO_PER_CLICK_dict.keys():
                        NOO_PER_CLICK_dict[k] += noo_per_click[k]


        results_dict = {
            'NoC@50': sum(NOC[0.5])/sum(NOO[0.5]),
            'NoC@65': sum(NOC[0.65])/sum(NOO[0.65]),
            'NoC@80': sum(NOC[0.8])/sum(NOO[0.8]),
            'NoC@85': sum(NOC[0.85])/sum(NOO[0.85]),
            'NoC@90': sum(NOC[0.9])/sum(NOO[0.9]),
            'IoU@1': IOU_PER_CLICK_dict['1']/NOO_PER_CLICK_dict['1'],
            'IoU@2': IOU_PER_CLICK_dict['2']/NOO_PER_CLICK_dict['2'],
            'IoU@3': IOU_PER_CLICK_dict['3']/NOO_PER_CLICK_dict['3'],
            'IoU@5': IOU_PER_CLICK_dict['5']/NOO_PER_CLICK_dict['5'],
            'IoU@10': IOU_PER_CLICK_dict['10']/NOO_PER_CLICK_dict['10'],
            'IoU@15': IOU_PER_CLICK_dict['15']/NOO_PER_CLICK_dict['15']
        }
        print('****************************')
        print(results_dict)

        return results_dict
