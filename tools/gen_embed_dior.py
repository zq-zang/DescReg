from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

categories = ['airplane', 'baseballfield', 'bridge', 'chimney',
              'dam', 'expressway service area', 'expressway toll station',
              'golffield', 'harbor', 'overpass', 'ship', 'stadium',
              'storagetank', 'tenniscourt', 'train station', 'vehicle',
              # unseen
              'airport', 'basketballcourt', 'groundtrackfield', 'windmill']

categories_semantic_descriptions = [
    "A flying vehicle used to transport people and goods long distances.",
    "An outdoor area where baseball games are played, typically with bases and a pitcher's mound.",
    "A structure that spans over an obstacle, like a river, to allow passage.",
    "A vertical structure, usually made of bricks, that allows smoke from a fire to escape.",
    "A barrier built across a river to store water, often for electricity generation.",
    "A designated spot along a highway where drivers can rest, refuel, or get food.",
    "A point on a highway where drivers pay a fee to continue using the road.",
    "A large area of grass where the game of golf is played, featuring holes, tees, and often sand traps.",
    "A sheltered area of water where ships can dock or anchor safely.",
    "A bridge-like structure allowing one road to cross over another without intersecting.",
    "A large boat or vessel designed to travel on water, transporting goods or people.",
    "A large venue with seats for spectators, typically used for sports or concerts.",
    "A large container, often cylindrical, for holding liquids or gases.",
    "A flat, rectangular area where tennis is played, marked with lines and often fenced in.",
    "A place where trains pick up or drop off passengers or goods.",
    "A means of transportation, like cars or trucks, that travel on roads.",
    "A facility where airplanes take off and land, often with terminals for passengers.",
    "A flat, rectangular area where basketball is played, marked with hoops at each end.",
    "An outdoor area designed for track and field events, like running, jumping, or throwing.",
    "A structure that uses wind to produce power, often seen with rotating blades."
]



categories_visual_descriptions = {
    'airplane': 'Flies in the sky, transporting people and goods.',
    'baseballfield': 'Designed for baseball games with bases arranged in a diamond pattern.',
    'bridge': 'Spans over water or another obstacle to connect two areas.',
    'chimney': 'Releases smoke from a fireplace or furnace in buildings.',
    'dam': 'A barrier across a river controlling water flow, often used for electricity generation.',
    'expressway service area': 'A rest stop along a highway for travelers to park, eat, and use facilities.',
    'expressway toll station': 'Where vehicles pay a fee on a highway to continue their journey.',
    'golffield': 'An open area for golf games with holes, greens, and fairways.',
    'harbor': 'Sheltered water area for ships and boats to dock or anchor.',
    'overpass': 'Allows one road to pass over another, typically in urban areas.',
    'ship': 'Large water vessel for transporting goods or passengers over seas.',
    'stadium': 'Venue with seating for spectators to watch sports or events.',
    'storagetank': 'Container for storing liquids or gases, commonly found in industrial zones.',
    'tenniscourt': 'Flat, rectangular surface for tennis with a central net.',
    'train station': 'Facility where trains pick up and drop off passengers, featuring platforms and tracks.',
    'vehicle': 'Means of transport, like cars or trucks, that travel on roads.',
    'airport': 'Facility where airplanes take off and land, and where passengers board flights.',
    'basketballcourt': 'Designed for basketball with a hoop at each end.',
    'groundtrackfield': 'Athletic field for track and field events such as running or jumping.',
    'windmill': 'Structure with rotating blades that harnesses wind energy for power or water pumping.'
}

# please decribe how these objects look like from a remote sensing image, please focus on visual appearance and use word in BERT vocabulary, please return python list
categories_visual_descriptions_new = [
    "A metallic object with wings extending from the center, typically appears as a small, elongated feature, especially if in-flight. On the ground, it can be surrounded by linear or organized patterns, indicating runways or taxiways.",
    "A diamond-shaped field with a noticeable pitcher's mound in the center, often surrounded by a larger oval or circular shape representing the outfield.",
    "Linear structures spanning over bodies of water or valleys, may cast shadows depending on the sun's angle.",
    "Tall, slender structures, often cylindrical, emitting a plume or shadow. Usually found adjacent to industrial buildings.",
    "A thick, linear structure often found across a river or stream, downstream side might show a reservoir.",
    "Clusters of structures next to an expressway, often with parking areas and sometimes fuel stations.",
    "Structures or booths aligned in a row adjacent to an expressway, usually with multiple lanes converging towards them.",
    "A vast green area interspersed with smaller sand patches (sand traps) and occasional water hazards. Green putting areas are also noticeable.",
    "Coastal or shoreline areas filled with anchored boats/ships, often surrounded by piers, wharfs, or jetties.",
    "Elevated roads crossing over other roads or railways, may cast shadows underneath.",
    "Objects floating on water, usually elongated, sometimes with a visible wake behind them if moving.",
    "Large oval or circular structures, often with a central open area. Depending on resolution, rows or sections might be visible.",
    "Large circular structures, often found in clusters, especially in industrial areas.",
    "Rectangular fields divided into two equal halves by a central net, often bright in color compared to surroundings.",
    "Clustered structures adjacent to parallel linear features (railway tracks). Often accompanied by platforms and parked trains.",
    "Small, colored or metallic spots on roads or parking areas.",
    "A vast area with long, straight runways. Large terminals, parked airplanes, and taxiways can also be visible.",
    "Rectangular fields with a visible center circle and two smaller circles at either end.",
    "Oval tracks with a grassy interior, which may contain fields or courts.",
    "Tall, slender structures with long, linear blades radiating from a central hub. Often found in groups in open areas."
]


categories_visual_descriptions_simplified = [
    "Linear shape with wings on sides and a tail at the rear.", # airplane
    "Diamond shape with white lines, lighter infield than outfield.", # baseballfield
    "Linear structure over water or land, may have shadows beneath.", # bridge
    "Tall, thin cylinder, possibly releasing smoke or steam.", # chimney
    "Wall-like structure across water with different water levels on each side.", # dam
    "Cluster of buildings by a road with parking spots.", # expressway service area
    "Linear structure on road with small booths and nearby vehicles.", # expressway toll station
    "Open areas with sand pits and ponds, small circles for greens.", # golffield
    "Docks and piers by water with boat shapes nearby.", # harbor
    "Elevated road crossing another, with shadows below.", # overpass
    "Elongated shape on water, possibly with rising structures.", # ship
    "Large oval or circle with an open center and tiered look.", # stadium
    "Circular shape, often with a dome top and casting shadows.", # storagetank
    "Light-colored rectangle with a mid-line.", # tenniscourt
    "Parallel lines with platforms and larger buildings nearby.", # train station
    "Small dot on roads, varying size based on type.", # vehicle
    "Large open areas with runways, cluster of buildings, airplane shapes nearby.", # airport
    "Rectangle with markings at ends, uniform color.", # basketballcourt
    "Oval track around an inner field, different colors.", # groundtrackfield
    "Tall structure with long blades radiating from the center." # windmill
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
embeddings1 = model.encode(categories_semantic_descriptions, convert_to_tensor=True)
embeddings = embeddings1 / np.linalg.norm(embeddings1, ord=2, axis=1, keepdims=True)
bg_embedding = np.random.rand(1, embeddings1.shape[1])
full_embeddings = np.concatenate([bg_embedding, embeddings1])
# np.save('xview_background_seen_and_unseen_word_vec_original_sentence_transformer.npy', full_embeddings)

# embeddings1 = np.load('data/visdrone/annotations/background_seen_and_unseen_word_vec.npy')[1:,]
# embeddings = embeddings1 / np.linalg.norm(embeddings1, ord=2, axis=1, keepdims=True)

similarity_matrix = cosine_similarity(embeddings)
sim_matrix_xview_cosine = expand_similarity_matrix(similarity_matrix)
np.save('sim_matrix_dior_cosine.npy', sim_matrix_xview_cosine)
similarity_matrix_normalized = softmax((similarity_matrix-np.eye(20))/temperature, axis=1)
similarity_matrix_normalized += np.eye(20)
# add background dimensioin
sim_matrix_dior_normalized = expand_similarity_matrix(similarity_matrix_normalized)
np.save('sim_matrix_dior_normalized_{}.npy'.format(temperature), sim_matrix_dior_normalized)
distance_matrix = 1 - cosine_similarity(embeddings)
linkage_matrix = linkage(distance_matrix, method='average')


# Plot the dendrogram
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix, labels=categories)
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(15, 15))
# plt.imshow(sim_matrix_dior_normalized, cmap='hot', vmin=0, vmax=1, aspect='auto')
# plt.colorbar(label='Cosine Similarity')
# plt.xticks(range(len(categories)+1), ['bg'] + categories, rotation='vertical')
# plt.yticks(range(len(categories)+1), ['bg'] + categories)
# plt.tight_layout()
# plt.show()

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample data
# embeddings = np.random.rand(C, D)
# class_names = ["Class_" + str(i) for i in range(C)]

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=np.arange(reduced_data.shape[0]), cmap='jet', edgecolor='k')  # Using class indices as colors

for i, name in enumerate(categories):
    plt.annotate(name, (reduced_data[i, 0], reduced_data[i, 1]))

plt.colorbar()
plt.title('2D PCA of Class Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

print("Explained variance by component: ", pca.explained_variance_ratio_)
print("Total explained variance: ", sum(pca.explained_variance_ratio_))






