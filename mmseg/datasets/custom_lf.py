import os.path as osp
import random
from functools import reduce

import numpy as np
from terminaltables import AsciiTable
from torch.utils.data import Dataset

import mmcv
from mmcv.utils import print_log
from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class CustomLFDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.png',
                 ann_dir=None,
                 seg_map_suffix='.npy',
                 sai_dir=None,
                 sai_suffix='.png',
                 lf_dir=None,
                 lf_suffix='.png',
                 sai_number=5,
                 lf_number=25,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        
        
        self.pipeline = Compose(pipeline)   
        self.img_dir = img_dir + split + '/' 
        self.img_suffix = img_suffix    
        self.ann_dir = ann_dir + split + '/' 
        self.seg_map_suffix = seg_map_suffix   
        self.sai_dir = sai_dir + split + '/'  
        self.sai_suffix = sai_suffix  
        
        self.lf_dir = lf_dir + split + '/'  
        self.lf_suffix = lf_suffix  
        
        self.sai_number = sai_number    
        self.lf_number = lf_number      
        
        
        self.split = split     
        self.data_root = data_root  
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)  
        
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)


    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def get_sequence_info(self, idx):
        """Get sequence by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Sequence image info of specified index.
        """

        return self.img_infos[idx]['sai_sequence']

    def get_sequence_index_info(self, idx):
        return self.img_infos[idx]['sai_sequence_index']
        
    def get_lf_info(self, idx):
        return self.img_infos[idx]['lf_sequence']
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['sequence_prefix'] = self.sai_dir
        results['lf_prefix'] = self.lf_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]  
        ann_info = self.get_ann_info(idx) 
        sequence_info = self.get_sequence_info(idx)
        lf_info = self.get_lf_info(idx)
        sequence_index = self.get_sequence_index_info(idx) 
        results = dict(img_info=img_info, ann_info=ann_info, sequence_info=sequence_info,lf_info = lf_info, sequence_index=sequence_index)  
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        img_info = self.img_infos[idx]
        sequence_info = self.get_sequence_info(idx)
        sequence_index = self.get_sequence_index_info(idx) 
        lf_info = self.get_lf_info(idx)
        image_list = sequence_info["sais"][0].split('/')[0]
        
        total_result = []
        data_path = self.img_dir
        data_type = data_path.split('/')[-3]
        if self.sai_number == 5:
            sequence_info["sais"] = [image_list+'/5_1.png', image_list+'/5_2.png', image_list+'/5_3.png', image_list+'/5_4.png']
            sequence_index["sais_index"] = [[5,1],[5,2],[5,3],[5,4]]
        elif self.sai_number == 4:
            if data_type[12:15] == "big":   # For UrbanLF-Syn-Big-dis dataset, choose three reference views that have short distance
                sequence_info["sais"] = [image_list+'/5_2.png', image_list+'/5_3.png', image_list+'/5_4.png']
                sequence_index["sais_index"] = [[5,2],[5,3],[5,4]]
            else:       # For UrbanLF-Syn-Small-dis/UrbanLF-Real dataset, choose three reference views that have long distance
                sequence_info["sais"] = [image_list+'/5_1.png', image_list+'/5_2.png', image_list+'/5_3.png']
                sequence_index["sais_index"] = [[5,1],[5,2],[5,3]]
        results = dict(img_info=img_info, sequence_info=sequence_info,lf_info = lf_info,sequence_index=sequence_index)
        self.pre_pipeline(results)
        total_result.append(self.pipeline(results))
        
        
        if self.sai_number == 5:
            sequence_info["sais"] = [image_list+'/5_9.png', image_list+'/5_8.png', image_list+'/5_7.png', image_list+'/5_6.png']
            sequence_index["sais_index"] = [[5,9],[5,8],[5,7],[5,6]]
        elif self.sai_number == 4:
            if data_type[12:15] == "big":
                sequence_info["sais"] = [ image_list+'/5_8.png', image_list+'/5_7.png', image_list+'/5_6.png']
                sequence_index["sais_index"] = [[5,8],[5,7],[5,6]]
            else:
                sequence_info["sais"] = [ image_list+'/5_9.png', image_list+'/5_8.png', image_list+'/5_7.png']
                sequence_index["sais_index"] = [[5,9],[5,8],[5,7]]
        results = dict(img_info=img_info, sequence_info=sequence_info,lf_info = lf_info,sequence_index=sequence_index)
        self.pre_pipeline(results)
        total_result.append(self.pipeline(results))
        
        
        if self.sai_number == 5:
            sequence_info["sais"] = [image_list+'/1_5.png', image_list+'/2_5.png', image_list+'/3_5.png',image_list+'/4_5.png']
            sequence_index["sais_index"] = [[1,5],[2,5],[3,5],[4,5]]
        elif self.sai_number == 4:
            if data_type[12:15] == "big":
                sequence_info["sais"] = [image_list+'/2_5.png', image_list+'/3_5.png',image_list+'/4_5.png']
                sequence_index["sais_index"] = [[2,5],[3,5],[4,5]]
            else:
                sequence_info["sais"] = [image_list+'/1_5.png', image_list+'/2_5.png',image_list+'/3_5.png']
                sequence_index["sais_index"] = [[1,5],[2,5],[3,5]]
        results = dict(img_info=img_info, sequence_info=sequence_info,lf_info = lf_info,sequence_index=sequence_index)
        self.pre_pipeline(results)
        total_result.append(self.pipeline(results))
        
        
        if self.sai_number == 5:
            sequence_info["sais"] = [image_list+'/9_5.png', image_list+'/8_5.png', image_list+'/7_5.png', image_list+'/6_5.png']
            sequence_index["sais_index"] = [[9,5],[8,5],[7,5],[6,5]]
        elif self.sai_number == 4:
            if data_type[12:15] == "big":
                sequence_info["sais"] = [image_list+'/8_5.png', image_list+'/7_5.png', image_list+'/6_5.png']
                sequence_index["sais_index"] = [[8,5],[7,5],[6,5]]
            else:
                sequence_info["sais"] = [image_list+'/9_5.png', image_list+'/8_5.png', image_list+'/7_5.png']
                sequence_index["sais_index"] = [[9,5],[8,5],[7,5]]
        results = dict(img_info=img_info, sequence_info=sequence_info,lf_info = lf_info,sequence_index=sequence_index)
        self.pre_pipeline(results)
        total_result.append(self.pipeline(results))
         
        if self.sai_number == 5:
            sequence_info["sais"] = [image_list+'/1_1.png', image_list+'/2_2.png', image_list+'/3_3.png', image_list+'/4_4.png']
            sequence_index["sais_index"] = [[1,1],[2,2],[3,3],[4,4]]
        elif self.sai_number == 4:
            if data_type[12:15] == "big":
                sequence_info["sais"] = [ image_list+'/2_2.png', image_list+'/3_3.png', image_list+'/4_4.png']
                sequence_index["sais_index"] = [[2,2],[3,3],[4,4]]
            else:
                sequence_info["sais"] = [ image_list+'/1_1.png', image_list+'/2_2.png', image_list+'/3_3.png']
                sequence_index["sais_index"] = [[1,1],[2,2],[3,3]]
        results = dict(img_info=img_info, sequence_info=sequence_info,lf_info = lf_info,sequence_index=sequence_index)
        self.pre_pipeline(results)
        total_result.append(self.pipeline(results))
        
        if self.sai_number == 5:
            sequence_info["sais"] = [image_list+'/9_9.png', image_list+'/8_8.png', image_list+'/7_7.png', image_list+'/6_6.png']
            sequence_index["sais_index"] = [[9,9],[8,8],[7,7],[6,6]]
        elif self.sai_number == 4:
            if data_type[12:15] == "big":
                sequence_info["sais"] = [ image_list+'/8_8.png', image_list+'/7_7.png', image_list+'/6_6.png']
                sequence_index["sais_index"] = [[8,8],[7,7],[6,6]]
            else:
                sequence_info["sais"] = [ image_list+'/9_9.png', image_list+'/8_8.png', image_list+'/7_7.png']
                sequence_index["sais_index"] = [[9,9],[8,8],[7,7]]
        results = dict(img_info=img_info, sequence_info=sequence_info,lf_info = lf_info,sequence_index=sequence_index)
        self.pre_pipeline(results)
        total_result.append(self.pipeline(results))
        
        
        if self.sai_number == 5:
            sequence_info["sais"] = [image_list+'/1_9.png', image_list+'/2_8.png', image_list+'/3_7.png', image_list+'/4_6.png']
            sequence_index["sais_index"] = [[1,9],[2,8],[3,7],[4,6]]
        elif self.sai_number == 4:
            if data_type[12:15] == "big":
                sequence_info["sais"] = [image_list+'/2_8.png', image_list+'/3_7.png', image_list+'/4_6.png']
                sequence_index["sais_index"] = [[2,8],[3,7],[4,6]]
            else:
                sequence_info["sais"] = [image_list+'/1_9.png', image_list+'/2_8.png', image_list+'/3_7.png']
                sequence_index["sais_index"] = [[1,9],[2,8],[3,7]]
        results = dict(img_info=img_info, sequence_info=sequence_info,lf_info = lf_info,sequence_index=sequence_index)
        self.pre_pipeline(results)
        total_result.append(self.pipeline(results))
        
        if self.sai_number == 5:
            sequence_info["sais"] = [image_list+'/9_1.png', image_list+'/8_2.png', image_list+'/7_3.png', image_list+'/6_4.png']
            sequence_index["sais_index"] = [[9,1],[8,2],[7,3],[6,4]]
        elif self.sai_number == 4:
            if data_type[12:15] == "big":
                sequence_info["sais"] = [ image_list+'/8_2.png', image_list+'/7_3.png', image_list+'/6_4.png']
                sequence_index["sais_index"] = [[8,2],[7,3],[6,4]]
            else:
                sequence_info["sais"] = [ image_list+'/9_1.png', image_list+'/8_2.png', image_list+'/7_3.png']
                sequence_index["sais_index"] = [[9,1],[8,2],[7,3]]
        results = dict(img_info=img_info, sequence_info=sequence_info,lf_info = lf_info,sequence_index=sequence_index)
        self.pre_pipeline(results)
        total_result.append(self.pipeline(results))
          
        return total_result

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            gt_seg_map = np.load(seg_map) - 1
            #gt_seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            ignore_index=self.ignore_index,
            metrics=metric)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            #np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
            ret_metric * 100 for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            #np.round(np.nanmean(ret_metric) * 100, 2)
            np.nanmean(ret_metric) * 100
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
            [i]] = summary_table_data[1][i] / 100.0
        return eval_results
