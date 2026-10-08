import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json


def visualize_dataset(dataset_file, image_dir):
    # Load dataset
    with open(dataset_file, 'r') as f:
        data = json.load(f)

    # Create a dictionary for easy lookup
    img_dict = {img['id']: img for img in data['images']}
    cat_dict = {cat['id']: cat for cat in data['categories']}
    anno_dict = {img['id']: [] for img in data['images']}

    for anno in data['annotations']:
        anno_dict[anno['image_id']].append(anno)
        # anno_dict[anno['image_id'] - 1].append(anno)# if use dior

    show_cats = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle',
               'small-vehicle', 'roundabout',
               'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool']
    show_cats = ["fixed wing aircraft", "small aircraft", "passenger plane or cargo plane", "helicopter", "passenger vehicle", "small car", "bus",
                "pickup truck", "utility truck", "truck", "cargo truck", "truck tractor with box trailer", "truck tractor", "trailer", "truck tractor with flatbed trailer",
                "truck tractor with liquid tank", "crane truck", "railway vehicle", "passenger car", "cargo car or container car", "flat car", "tank car", "locomotive",
                "maritime vessel", "motorboat", "sailboat", "tugboat", "barge", "fishing vessel", "ferry", "yacht", "container ship", "oil tanker", "engineering vehicle",
                "tower crane", "container crane", "reach stacker", "straddle carrier", "mobile crane", "dump truck", "haul truck", "scraper or tractor", "front loader or bulldozer",
                "excavator", "cement mixer", "ground grader", "hut or tent", "shed", "building", "aircraft hangar", "damaged building", "facility", "construction site", "vehicle lot",
                "helipad", "storage tank", "shipping container lot", "shipping container", "pylon", "tower"]
    # Iterate through images
    cls = 1
    count = 0
    # unseen_class_ids = [3, 6, 7, 11, 23, 24, 27, 36, 38, 41, 43, 56]
    unseen_class_ids = [16, 17, 18, 19]
    for img_id, img_info in img_dict.items():
        # if img_id>=300:
        # print('xx')
        # continue
        cats_in_img = set([a['category_id'] for a in anno_dict[img_id]])
        # cats_in_img_within_unseen = bool(cats_in_img & set([i for i in range(40) if i not in unseen_class_ids]))
        cats_in_img_within_unseen = bool(cats_in_img & set(list(range(16))))
        # cats_in_img_within_unseen = bool(cats_in_img & set(self.unseen_class_ids))
        # cats_in_img_within_unseen = bool(cats_in_img & set([]))
        if cats_in_img_within_unseen or len(cats_in_img) == 0:
             continue
        # count+=1
        # if count == 4:
        #     cls+=1
        #     count=0
        img = cv2.imread(f"{image_dir}/{img_info['file_name']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(img)

        # Draw bounding boxes
        for anno in anno_dict[img_id]:
            # Create a Rectangle patch
            rect = patches.Rectangle((anno['bbox'][0], anno['bbox'][1]),
                                     anno['bbox'][2], anno['bbox'][3],
                                     linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add label
            cat_id = anno['category_id']
            plt.text(anno['bbox'][0], anno['bbox'][1] - 2,
                     cat_dict[cat_id]['name'],
                     color='red')

        # Show the figure with a block that will be unblocked on close event
        plt.savefig('./data/dior/test_detimg/'+'gt_'+img_info['file_name'])
        # plt.show(block=True)



# dataset_file = "./data/visdrone/annotations/validation.json"  # replace with your json annotations file
# image_dir = "./data/visdrone/train_test"  # replace with the path to your images
dataset_file = "./data/dior/Dior_test_anno.json"  # replace with your json annotations file
image_dir = "./data/dior/test/"  # replace with the path to your images
# dataset_file = "./data/dota/val/val.json"  # replace with your json annotations file
# image_dir = "./data/dota/val/val_images"  # replace with the path to your images
# dataset_file = "./data/xview/val_anno.json"  # replace with your json annotations file
# image_dir = "./data/xview/val"  # replace with the path to your images
visualize_dataset(dataset_file, image_dir)
