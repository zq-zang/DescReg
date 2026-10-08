from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

categories = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
    'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
    'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']


categories_visual_descriptions = [
    "A silhouette or figure with a head, torso, two arms, and two legs.", # person
    "Two circular wheels connected by a frame with handlebars at the front.", # bicycle
    "Four circular wheels under a metal frame, often with visible windows and doors.", # car
    "Two wheels, one behind the other, with handlebars and often a visible seat.", # motorcycle
    "A long fuselage with wings extending from the sides and a tail fin.", # airplane
    "Large rectangular box shape, typically with multiple windows and a set of wheels.", # bus
    "Multiple connected rectangular shapes, often on tracks, with windows.", # train
    "Large rectangular shape, front cab with windows, often with a larger cargo area at the back.", # truck
    "Curved bottom shape, often with a pointed front (bow).", # boat
    "Vertical post with three circular lights: red, yellow, and green.", # traffic_light
    "Short, stout cylindrical shape, often in red or yellow.", # fire_hydrant
    "Red octagon with white edges.", # stop_sign
    "Tall, slender post with a box on top.", # parking_meter
    "Horizontal plank supported by two or more legs or supports.", # bench
    "Small, often oval shape with wings extending outwards and a pointed beak.", # bird
    "Four-legged silhouette with a tail, pointy ears, and a rounded face.", # cat
    "Four-legged silhouette with a tail, varied ear shapes, and snout.", # dog
    "Large four-legged silhouette, long neck, tail, and often with a mane.", # horse
    "Rounded, fluffy silhouette on four legs with a head jutting out.", # sheep
    "Large four-legged silhouette, often with visible udders and horns.", # cow
    "Huge silhouette, long trunk extending from face, big floppy ears, four legs.", # elephant
    "Stout four-legged figure, round face, and often a short tail.", # bear
    "Four-legged silhouette, with black and white stripes.", # zebra
    "Extremely long neck, four legs, and small horn-like \"ossicones\" on the head.", # giraffe
    "Rounded or rectangular shape with two shoulder straps.", # backpack
    "Semi-circular canopy with a central rod underneath.", # umbrella
    "Rectangular or oval shape, often with handles or a strap.", # handbag
    "Long, thin, downward-pointing triangle.", # tie
    "Rectangular box with a handle, often with latches.", # suitcase
    "Flat, perfect circle.", # frisbee
    "Two long, thin rectangles, often seen in parallel.", # skis
    "Single long, narrow, slightly curved rectangle.", # snowboard
    "Perfect circle, may have pattern or lacing depending on the sport.", # sports_ball
    "Triangular or diamond shape, often with tail streamers.", # kite
    "Long, slightly tapered cylindrical shape.", # baseball_bat
    "Rounded shape with finger-like extensions.", # baseball_glove
    "Flat, elongated oval with four small wheels underneath.", # skateboard
    "Long, narrow, slightly curved flat shape.", # surfboard
    "Oval with a mesh center and a handle extending downward.", # tennis_racket
    "Cylindrical shape with a narrow top.", # bottle
    "Stem with an inverted conical top.", # wine_glass
    "Cylindrical shape with a handle, often open at the top.", # cup
    "Handle with several pointed tines extending from it.", # fork
    "Flat, thin blade extending from a handle.", # knife
    "Handle with a small, often rounded, bowl shape at the end.", # spoon
    "Semi-circular or deep oval shape.", # bowl
    "Curved, elongated shape.", # banana
    "Rounded shape, sometimes with a dimple at the top and bottom.", # apple
    "Two flat squares with varied colored layers in between.", # sandwich
    "Perfect circle with textured surface.", # orange
    "Tree-like green clusters.", # broccoli
    "Long, tapered shape, often orange.", # carrot
    "Elongated shape with a bun on the outside and a cylindrical shape (sausage) inside.", # hot_dog
    "Large circle, often with colorful toppings scattered.", # pizza
    "Circle with a hole in the middle.", # donut
    "Round or rectangular shape, often with layers and icing.", # cake
    "Four legs, a seat, and often a backrest.", # chair
    "Longer seat, typically with cushions, backrest, and armrests.", # couch
    "Circular shape (pot) with greenery protruding above.", # potted_plant
    "Large rectangle, often with pillows and a blanket.", # bed
    "Flat rectangle or circle supported by legs.", # dining_table
    "U or oval-shaped bowl with a lid, often with a tank behind.", # toilet
    "Rectangle, often emitting light or displaying images.", # tv
    "Thin rectangle, can be seen opened to reveal a screen and keyboard.", # laptop
    "Small, rounded shape with a wire or possibly wireless.", # mouse
    "Thin rectangle, often with buttons.", # remote
    "Flat rectangle with multiple small rectangular or square buttons.", # keyboard
    "Thin rectangle, often with a lighted screen.", # cell_phone
    "Box shape with a window on one side.", # microwave
    "Larger box shape, often with a front door.", # oven
    "Box shape with slots on the top.", # toaster
    "Concave shape, often surrounded by a flat counter.", # sink
    "Tall box, often with one or two doors.", # refrigerator
    "Rectangle with a visible spine, and possibly visible pages.", # book
    "Often circular with numbers and two or three hands pointing at them.", # clock
    "Tall, often cylindrical or hourglass shape.", # vase
    "Two elongated blades pivoted together with handles.", # scissors
    "Bear-like figure, often seated, with round ears and limbs.", # teddy_bear
    "Handle with a nozzle pointing outward.", # hair_drier
    "Long handle with a head covered in bristles." # toothbrush
]


def expand_similarity_matrix(sim_mat):
    # Check if input matrix is square
    if sim_mat.shape[0] != sim_mat.shape[1]:
        raise ValueError("The input similarity matrix must be square.")

    # Create a new (C+1)x(C+1) matrix filled with zeros
    new_sim_mat = np.zeros((sim_mat.shape[0]+1, sim_mat.shape[1]+1))

    # Copy the original matrix into the bottom right corner of the new matrix
    new_sim_mat[1:, 1:] = sim_mat

    # Set the first row and first column to 0 (optional, already done by np.zeros)
    new_sim_mat[0, :] = 0
    new_sim_mat[:, 0] = 0

    # Set the similarity of the new class to itself to 1
    new_sim_mat[0, 0] = 1

    return new_sim_mat

temperature = 0.03

model = SentenceTransformer('all-MiniLM-L6-v2')
#Compute embedding for both lists
embeddings1 = model.encode(categories_visual_descriptions, convert_to_tensor=True)
embeddings = embeddings1 / np.linalg.norm(embeddings1, ord=2, axis=1, keepdims=True)
bg_embedding = np.random.rand(1, embeddings1.shape[1])
full_embeddings = np.concatenate([bg_embedding, embeddings1])
# np.save('xview_background_seen_and_unseen_word_vec_original_sentence_transformer.npy', full_embeddings)

similarity_matrix = cosine_similarity(embeddings)
sim_matrix_xview_cosine = expand_similarity_matrix(similarity_matrix)
np.save('sim_matrix_coco_cosine.npy', sim_matrix_xview_cosine)
similarity_matrix_normalized = softmax((similarity_matrix-np.eye(embeddings.size(0)))/temperature, axis=1)
similarity_matrix_normalized += np.eye(embeddings.size(0))
# add background dimensioin
sim_matrix_coco_normalized = expand_similarity_matrix(similarity_matrix_normalized)
np.save('sim_matrix_coco_normalized_{}.npy'.format(temperature), sim_matrix_coco_normalized)
distance_matrix = 1 - cosine_similarity(embeddings)
linkage_matrix = linkage(distance_matrix, method='average')


# Plot the dendrogram
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix, labels=categories)
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 15))
plt.imshow(sim_matrix_coco_normalized, cmap='hot', vmin=0, vmax=1, aspect='auto')
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(len(categories)+1), ['bg'] + categories, rotation='vertical')
plt.yticks(range(len(categories)+1), ['bg'] + categories)
plt.tight_layout()
plt.show()
