from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

categories = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','cat',
    'chair', 'cow', 'diningtable', 'horse', 'motorbike','person',
    'pottedplant', 'sheep', 'tvmonitor','car', 'dog', 'sofa', 'train']

categories_visual_descriptions = [
    "A streamlined structure with wings extending on either side, often with a tail section and engines or propellers. Usually larger than other objects in the image unless pictured from a great distance.",
    "Two wheels of the same size, one behind the other, connected by a frame. There are handlebars for steering at the front and a seat towards the rear.",
    "A small to medium-sized creature with wings extended from its sides, often with a distinct beak and feathered body. Might be captured in flight or perched.",
    "A water vessel, ranging from a small canoe shape to a larger structure. Typically seen floating on water, with the potential for sails, masts, or a streamlined shape.",
    "A cylindrical container with a narrow neck. Can be made of glass or plastic, often with a label around its middle.",
    "A large, elongated vehicle with multiple windows on its side. Often larger than most other vehicles in an image and can have a flat front or rounded front.",
    "A four-legged creature with a tail, often with pointed ears and whiskers. The size can vary but typically smaller than many other domestic animals.",
    "A structure with a seat, typically with four legs, and often with a backrest. Can vary in size and style but generally meant for a person to sit on.",
    "A large four-legged animal, often white with black or brown spots. Features include large ears, a tail, and might have horns.",
    "A flat horizontal surface, often supported by four legs. Generally larger than a chair and seen with objects on top like dishes or food.",
    "A tall four-legged animal with a long mane and tail. Often larger than many other animals and can be seen standing or in motion.",
    "Two-wheeled vehicle, smaller than a car, with a seat for the rider. Often has handlebars and might have a distinct shape for the gas tank in front of the seat.",
    "A two-legged figure, typically with a head on top, two arms extending from the sides, and standing upright. Size can vary based on perspective in the image.",
    "A plant or a set of leaves rising from a container or pot. The container is typically round and might be made of clay, plastic, or other materials.",
    "A four-legged animal, often with a thick woolly coat. Size is medium, typically smaller than a cow but larger than a cat.",
    "A rectangular object, often with a flat and reflective screen on the front. Can be seen on a stand or mounted to a wall.",
    "A four-wheeled vehicle, typically smaller than a bus. Has windows around, and the shape can vary from compact to elongated.",
    "A four-legged creature with a tail, varying significantly in size and coat type. Features might include floppy or pointed ears, and a variety of facial structures.",
    "A large, cushioned seat, often big enough for multiple people. Can be seen with a backrest and armrests on the sides.",
    "A long, elongated vehicle often seen on tracks. Consists of multiple connected sections or carriages and can be seen from a side-view or front-view depending on perspective."
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
embeddings1 = model.encode(categories, convert_to_tensor=True)
embeddings = embeddings1 / np.linalg.norm(embeddings1, ord=2, axis=1, keepdims=True)
bg_embedding = np.random.rand(1, embeddings1.shape[1])
full_embeddings = np.concatenate([bg_embedding, embeddings1])
# np.save('xview_background_seen_and_unseen_word_vec_original_sentence_transformer.npy', full_embeddings)

similarity_matrix = cosine_similarity(embeddings)
sim_matrix_xview_cosine = expand_similarity_matrix(similarity_matrix)
np.save('sim_matrix_pascal_cosine.npy', sim_matrix_xview_cosine)
similarity_matrix_normalized = softmax((similarity_matrix-np.eye(embeddings.size(0)))/temperature, axis=1)
similarity_matrix_normalized += np.eye(embeddings.size(0))
# add background dimensioin
sim_matrix_pascal_normalized = expand_similarity_matrix(similarity_matrix_normalized)
np.save('sim_matrix_pascal_normalized_{}.npy'.format(temperature), sim_matrix_pascal_normalized)
distance_matrix = 1 - cosine_similarity(embeddings)
linkage_matrix = linkage(distance_matrix, method='average')


# Plot the dendrogram
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix, labels=categories)
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.show()
#

import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from ete3 import Tree, TreeStyle, NodeStyle

# Generate a random CxD matrix as an example (replace with your embeddings)
np.random.seed(42)
C, D = 10, 5
embeddings = np.random.rand(C, D)

# Compute pairwise distances
distances = pdist(embeddings)

# Perform hierarchical clustering
Z = linkage(distances, method='ward')


# Convert hierarchical clustering results to Newick format
def convert_to_newick(Z, labels):
    def build_tree(linkage, idx):
        if idx < len(labels):
            return labels[idx]
        left = int(linkage[idx - len(labels), 0])
        right = int(linkage[idx - len(labels), 1])
        return "(%s,%s)" % (build_tree(linkage, left), build_tree(linkage, right))

    return build_tree(Z, len(Z) + len(labels) - 1) + ';'


newick_str = convert_to_newick(Z, list(map(str, range(len(embeddings)))))

# Visualize using ete3
t = Tree(newick_str)
ts = TreeStyle()
ts.mode = "r"
ts.show_leaf_name = True
ts.scale = 120

# Set the appearance of nodes
nstyle = NodeStyle()
nstyle["size"] = 0
for n in t.traverse():
    n.set_style(nstyle)

t.show(tree_style=ts)

# import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(embeddings)
#
# plt.figure(figsize=(10, 7))
# # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=np.arange(reduced_data.shape[0]), cmap='jet', edgecolor='k')  # Using class indices as colors
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], cmap='jet', edgecolor='k')  # Using class indices as colors
#
# for i, name in enumerate(categories):
#     plt.annotate(name, (reduced_data[i, 0], reduced_data[i, 1]))
#
# # plt.colorbar()
# plt.title('2D PCA of Class Embeddings')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.show()
#
# print("Explained variance by component: ", pca.explained_variance_ratio_)
# print("Total explained variance: ", sum(pca.explained_variance_ratio_))
#





