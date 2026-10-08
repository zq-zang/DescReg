from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

categories =['airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
               'golffield', 'harbor',
               'overpass', 'ship', 'stadium', 'storagetank',
               'tenniscourt', 'trainstation', 'vehicle', 'airport', 'basketballcourt', 'groundtrackfield', 'windmill']
categories = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle',
               'small-vehicle', 'roundabout',
               'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool']
# categories = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 'ground-track-field', 'harbor', 'bridge',
#                 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
categories = [' '.join(c.split('-')) for c in categories]

# Load pre-trained model and tokenizer
# model_name = 'bert-large-uncased'
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Store all category embeddings here
category_embeddings = {}
category_embeddings_array = []

# Process each category
for category in categories:
    # Tokenize the category
    tokens = tokenizer.tokenize(category)

    # Convert tokens to input IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Add batch dimension and convert to tensor
    token_ids = torch.tensor([token_ids])

    # Generate embeddings
    with torch.no_grad():
        outputs = model(token_ids)

    # Extract the embeddings for the `[CLS]` token
    category_embedding = outputs[0][0, 0, :]

    # Store the category embeddings
    category_embeddings[category] = category_embedding
    category_embeddings_array.append(category_embedding)

orig_embeddings = np.stack(category_embeddings_array, 0)
bg_embedding = np.random.rand(1, orig_embeddings.shape[1])
full_embeddings = np.concatenate([bg_embedding, orig_embeddings])
# np.save('data/dota/dota_background_seen_and_unseen_word_vec.npy', full_embeddings)

embeddings = orig_embeddings / np.linalg.norm(orig_embeddings, ord=2, axis=1, keepdims=True)
# assuming `embeddings` is a 2D array of your embeddings, shape (21, 1024)
distance_matrix = 1 - cosine_similarity(embeddings)

# Use 'precomputed' because distance_matrix is a distance matrix, not raw data
# 'average' method can be replaced with 'single', 'complete', 'weighted', etc.
# depending on how you want to handle the distance between clusters
linkage_matrix = linkage(distance_matrix, method='average')

plt.figure(figsize=(10, 7))

# Plot the dendrogram
dendrogram(linkage_matrix, labels=categories)
plt.xticks(rotation=45)
plt.show()


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assume `embedding_matrix` is your category embedding of shape CxD
# And `category_names` is your list of category names
# For instance:
# embedding_matrix = np.random.rand(10, 200)
# category_names = ['cat1', 'cat2', ..., 'cat10']

tsne = TSNE(n_components=2, random_state=42)
embedding_2d = tsne.fit_transform(embeddings)

# Create a scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1])

# Annotate each point on the scatterplot with its respective category name:
for i in range(embeddings.shape[0]):
    plt.annotate(categories[i], (embedding_2d[i, 0], embedding_2d[i, 1]))

plt.title('t-SNE visualization of category embeddings')
plt.show()



from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot the embeddings
fig, ax = plt.subplots()
ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

# Add labels
for i, label in enumerate(categories):
    ax.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

plt.show()



from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

cosine_sim_matrix = cosine_similarity(embeddings)

# perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(cosine_sim_matrix)

# plot the results
plt.figure(figsize=(10, 10))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])

# optionally, if you have a list of category names, you can annotate the points
for i, category in enumerate(categories):
    plt.annotate(category, (tsne_results[i, 0], tsne_results[i, 1]))

plt.show()


from sklearn.cluster import DBSCAN

# define DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)

# apply DBSCAN
clusters = dbscan.fit_predict(embeddings)

# use t-SNE for 2D visualization
tsne_results = tsne.fit_transform(embeddings)

# plot the results with different colors for each cluster
plt.figure(figsize=(10, 10))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis')

# add a legend (optional)
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

# annotate the points (optional)
for i, category in enumerate(categories):
   plt.annotate(category, (tsne_results[i, 0], tsne_results[i, 1]))

plt.show()