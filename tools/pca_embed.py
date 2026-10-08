import numpy as np
from sklearn.decomposition import PCA

embeddings = np.load('./data/visdrone/annotations/background_seen_and_unseen_word_vec.npy')

# Create labels for your classes
labels = ['background', 'airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 'Expressway-Service-area',
          'Expressway-toll-station', 'golffield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
          'tenniscourt', 'trainstation', 'vehicle', 'airport', 'basketballcourt', 'groundtrackfield', 'windmill']

# Apply PCA
pca = PCA(n_components=16)
reduced_embeddings = pca.fit_transform(embeddings)
np.save('data/visdrone/annotations/reduced_background_seen_and_unseen_word_vec.npy', reduced_embeddings)

