import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the embeddings
embeddings = np.load('./data/visdrone/annotations/background_seen_and_unseen_word_vec.npy')

# Create labels for your classes
labels = ['background', 'airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 'Expressway-Service-area',
          'Expressway-toll-station', 'golffield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
          'tenniscourt', 'trainstation', 'vehicle', 'airport', 'basketballcourt', 'groundtrackfield', 'windmill']

# Apply PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot the embeddings
fig, ax = plt.subplots()
ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

# Add labels
for i, label in enumerate(labels):
    ax.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

plt.show()
