from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

categories = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle',
               'small-vehicle', 'roundabout',
              'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool']

# categories_semantic_descriptions = \
# [
# ]

categories_visual_descriptions = [
    "Small metallic object, often with wings. Shades and color differences can denote wings, fuselage, tail, and engines.",
    "Linear structure in water. Can cast long shadows in low-angle sunlight.",
    "Large cylindrical objects with rounded tops. Usually found in groups in industrial areas.",
    "Diamond-shaped with bright green field and lighter-colored diamond.",
    "Rectangular with potential hoop structures at each end. Surface color varies.",
    "Oval track surrounding a green infield. Track might be reddish or gray.",
    "Area with mix of water and land, with structures like docks, ships, and cranes.",
    "Linear structures spanning over water or valleys. May show supporting pillars.",
    "Distinguishable from small vehicles by size, shape, and shadow.",
    "Cars or small trucks. Might appear as small colorful or metallic rectangles.",
    "Circular or oval road structure with connecting roads. Center might have distinct features.",
    "Rectangular with a consistent color and a clear midline. Net might be visible.",
    "Compact shape, rotor blades might be seen as blur or distinct lines.",
    "Large rectangular green field with goalposts. Boundary and midline might be visible.",
    "Blue color, rectangular or irregular shapes. Surrounded by patios or decks."
]

categories_visual_descriptions_2 = [
    "Metallic, elongated object, often with wings and seen on runways or in-flight.",
    "Large, floating structure typically found on water bodies, often with elongated shape.",
    "Circular or oval large structures, often metallic, usually seen in industrial areas.",
    "Diamond-shaped field, typically with a distinct infield and outfield division.",
    "Rectangular space with clear markings, typically with two hoops on either end.",
    "Oval or circular track surrounding a grassy area, may contain field event locations.",
    "Waterfront area dense with ships, docks, and possibly infrastructure.",
    "Linear structure connecting two land masses, usually over water or depressions.",
    "Distinguishable, elongated structures on roads, larger than typical vehicles.",
    "Common, small structures on roads or parking areas, typical of cars.",
    "Circular road structure designed to regulate traffic, cars may be seen around.",
    "Rectangular field with a net in the middle, typically bright in color.",
    "Small, round object with rotor blades, often seen on helipads or in-flight.",
    "Large rectangular grassy area with goals at each end, often has distinct markings.",
    "Rectangular or irregular-shaped water-filled areas, typically blue, found in residential or recreational areas."
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

temperature = 0.05

model = SentenceTransformer('all-MiniLM-L6-v2')
#Compute embedding for both lists
embeddings1 = model.encode(categories_visual_descriptions_2, convert_to_tensor=True)
embeddings = embeddings1 / np.linalg.norm(embeddings1, ord=2, axis=1, keepdims=True)
bg_embedding = np.random.rand(1, embeddings1.shape[1])
full_embeddings = np.concatenate([bg_embedding, embeddings1])
# np.save('xview_background_seen_and_unseen_word_vec_original_sentence_transformer.npy', full_embeddings)

similarity_matrix = cosine_similarity(embeddings)
sim_matrix_xview_cosine = expand_similarity_matrix(similarity_matrix)
np.save('sim_matrix_dota_cosine.npy', sim_matrix_xview_cosine)
similarity_matrix_normalized = softmax((similarity_matrix-np.eye(embeddings.size(0)))/temperature, axis=1)
similarity_matrix_normalized += np.eye(embeddings.size(0))
# add background dimensioin
sim_matrix_xview_normalized = expand_similarity_matrix(similarity_matrix_normalized)
np.save('sim_matrix_dota_normalized_{}.npy'.format(temperature), sim_matrix_xview_normalized)
distance_matrix = 1 - cosine_similarity(embeddings)
linkage_matrix = linkage(distance_matrix, method='average')


# Plot the dendrogram
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix, labels=categories)
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 7))
plt.imshow(similarity_matrix_normalized, cmap='hot', vmin=0, vmax=1, aspect='auto')
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(len(categories)), ['bg'] + categories, rotation='vertical')
plt.yticks(range(len(categories)), ['bg'] + categories)
plt.tight_layout()
plt.show()
