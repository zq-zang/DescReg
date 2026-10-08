import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class Dior_16_4(CustomDataset):

    CLASSES = ('airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
               'golffield', 'harbor',
               'overpass', 'ship', 'stadium', 'storagetank',
               'tenniscourt', 'trainstation', 'vehicle', 'airport', 'basketballcourt', 'groundtrackfield', 'windmill')

    def load_annotations(self, ann_file):
        self.zsd_test=False
        self.unseen_class_ids = (16,17,18,19)
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.seen_cat2label = {}
        index = 1
        for cat_id in self.cat_ids:
            if cat_id not in self.unseen_class_ids:
                self.seen_cat2label[cat_id] = index
                index+=1
        self.unseen_cat2label = {
            cat_id: i + 1 + 16
            for i, cat_id in enumerate(self.unseen_class_ids)
        }
        # self.img_ids = self.coco.getImgIds()[:1000]
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id'] + 1 # fix for image id issue
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        # return self._parse_ann_info_nomaping(self.img_infos[idx], ann_info)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            # if not self.test_mode:
            #     # added for filtering images contain unseen classes during training
            #     img_id = self.img_infos[i]['id'] + 1  # fix for image id issue
            #     ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            #     ann_info = self.coco.loadAnns(ann_ids)
            #     cats_in_img = set([a['category_id'] for a in ann_info])
            #     cats_in_img_within_unseen = bool(cats_in_img & set(self.unseen_class_ids))
            #     if cats_in_img_within_unseen:
            #         continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _filter_imgs_zsd_test(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            # added for filtering images contain unseen classes during training
            img_id = self.img_infos[i]['id'] + 1
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            cats_in_img = set([a['category_id'] for a in ann_info])
            cats_in_img_within_unseen = bool(cats_in_img & set(list(range(16))))
            if cats_in_img_within_unseen:
                 continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if self.test_mode:
                if ann.get('ignore', False):
                    continue
                x1, y1, w, h = ann['bbox']
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                if ann.get('iscrowd', False):
                    gt_bboxes_ignore.append(bbox)
                else:
                    gt_bboxes.append(bbox)
                    if ann['category_id'] in self.seen_cat2label:
                        gt_labels.append(self.seen_cat2label[ann['category_id']])
                    else:
                        gt_labels.append(self.unseen_cat2label[ann['category_id']])
                    gt_masks_ann.append(ann['segmentation'])
            else:# filter unseen class labels
                if ann['category_id'] in self.seen_cat2label:
                    if ann.get('ignore', False):
                        continue
                    x1, y1, w, h = ann['bbox']
                    if ann['area'] <= 0 or w < 1 or h < 1:
                        continue
                    bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                    if ann.get('iscrowd', False):
                        gt_bboxes_ignore.append(bbox)
                    else:
                        gt_bboxes.append(bbox)
                        gt_labels.append(self.seen_cat2label[ann['category_id']])

                        gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def _parse_ann_info_nomaping(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(ann['category_id'])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann