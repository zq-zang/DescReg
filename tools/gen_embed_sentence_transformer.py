from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


categories = ['airplane', 'baseballfield', 'bridge', 'chimney',
              'dam', 'expressway service area', 'expressway toll station',
              'golffield', 'harbor', 'overpass', 'ship', 'stadium',
              'storagetank', 'tenniscourt', 'train station', 'vehicle',
              # unseen
              'airport', 'basketballcourt', 'groundtrackfield', 'windmill']

categories_description = \
['Streamlined body with two wings on each side.',
'Half-circle or oval shape surrounded by grass, typically yellow in color, with stands around it.',
'Generally straight or curved structure, usually gray or concrete in color.',
'Round cylindrical structure, usually white or gray in color.',
'Long and narrow shape with embankments on both sides, often in concrete gray, surrounded by lakes or rivers.',
'Irregular shape surrounded by parking spaces, buildings, and greenery.',
'Rectangular shape with lanes and toll booths.',
'A sprawling expanse of green grass, with winding fairways, often surrounded by trees.',
'Body of water with a curved shape.',
'Intersection of several roads with a bridge-like structure.',
'The most common shape is rectangular, commonly seen in water.',
'Circular or oval shape surrounded by stands or seating.',
'Typically a cylindrical shape, often silver or white in color.',
'Rectangular shape, typically with a green playing surface.',
'Rectangular or complex shape with platforms, tracks, and buildings.',
'The most common shape is rectangular, of various colors, commonly seen on land.',
'Rectangular shape, with buildings around.',
'Rectangular shape with basketball hoops and markings on the ground.',
'Rectangular or square shape, often in red clay, grass, or hard court, possibly with surrounding stands.',
'Multi-blade shape, often in white or silver color.'
]

categories_xview = ["fixed wing aircraft", "small aircraft", "passenger plane or cargo plane", "helicopter", "passenger vehicle", "small car", "bus",
                "pickup truck", "utility truck", "truck", "cargo truck", "truck tractor with box trailer", "truck tractor", "trailer", "truck tractor with flatbed trailer",
                "truck tractor with liquid tank", "crane truck", "railway vehicle", "passenger car", "cargo car or container car", "flat car", "tank car", "locomotive",
                "maritime vessel", "motorboat", "sailboat", "tugboat", "barge", "fishing vessel", "ferry", "yacht", "container ship", "oil tanker", "engineering vehicle",
                "tower crane", "container crane", "reach stacker", "straddle carrier", "mobile crane", "dump truck", "haul truck", "scraper or tractor", "front loader or bulldozer",
                "excavator", "cement mixer", "ground grader", "hut or tent", "shed", "building", "aircraft hangar", "damaged building", "facility", "construction site", "vehicle lot",
                "helipad", "storage tank", "shipping container lot", "shipping container", "pylon", "tower"]
name2id = {k:i for i, k in enumerate(categories_xview)}
unseen= ['truck tractor with box trailer', 'pickup truck', 'bus', 'reach stacker', 'motorboat', 'maritime vessel',
             'barge', 'shipping container lot', 'scraper or tractor', 'excavator', 'mobile crane', 'helicopter']
unseen_ids = [name2id[n] for n in unseen]
categories_description_xview = \
[   "Fixed Wing Aircraft: An airplane type that generates lift from its non-rotating wings.",
    "Small Aircraft: A compact airplane with limited seating capacity.",
    "Passenger Plane or Cargo Plane: Aircraft for passenger transport with seats and amenities, or for goods with large open spaces.",
    "Helicopter: Aircraft with rotating blades allowing various flight movements.",
    "Passenger Vehicle: A motorized vehicle used mainly for passenger transportation.",
    "Small Car: A compact, fuel-efficient vehicle with limited seating.",
    "Bus: A large vehicle that carries multiple passengers, often in public transport.",
    "Pickup Truck: A vehicle with an open cargo area for goods or equipment transport.",
    "Utility Truck: Specialized vehicle equipped for specific work or service.",
    "Truck: A heavy motor vehicle for goods or materials transport.",
    "Cargo Truck: A truck designed specifically for cargo transport.",
    "Truck Tractor with Box Trailer: A truck that pulls a box-shaped trailer for shipping goods.",
    "Truck Tractor: A powerful vehicle designed to pull a detachable load.",
    "Trailer: A non-motorized vehicle towed for transporting goods.",
    "Truck Tractor with Flatbed Trailer: A truck pulling a flatbed trailer for large, heavy items.",
    "Truck Tractor with Liquid Tank: A truck hauling tanks containing liquid or gas.",
    "Crane Truck: A truck with a crane for lifting and moving heavy objects.",
    "Railway Vehicle: A vehicle for railway tracks travel.",
    "Passenger Car: A railway vehicle for passenger transport.",
    "Cargo Car or Container Car: An enclosed railway car for goods transport.",
    "Flat Car: A railway car with a flat, open deck for goods transport.",
    "Tank Car: A railway car designed for liquids or gases.",
    "Locomotive: A powered train unit pulling attached cars on tracks.",
    "Maritime Vessel: A craft for water transportation.",
    "Motorboat: A boat propelled by an internal combustion engine.",
    "Sailboat: A boat propelled by sails.",
    "Tugboat: A sturdy boat guiding or pushing larger vessels.",
    "Barge: A flat-bottomed boat for freight, typically on canals and rivers.",
    "Fishing Vessel: A boat or ship used to catch fish.",
    "Ferry: A boat or ship for conveying passengers and goods across water.",
    "Yacht: A medium-sized boat used for leisure.",
    "Container Ship: A large cargo ship carrying goods in standard-sized containers.",
    "Oil Tanker: A large ship for transporting vast oil quantities.",
    "Engineering Vehicle: A vehicle for construction or engineering tasks.",
    "Tower Crane: A crane fixed to the ground for lifting heavy loads in construction.",
    "Container Crane: A crane for loading and unloading shipping containers from ships.",
    "Reach Stacker: A vehicle handling intermodal cargo containers in terminals or ports.",
    "Straddle Carrier: A freight-carrying vehicle that straddles its load.",
    "Mobile Crane: A crane mounted on a movable platform.",
    "Dump Truck: A truck with a tilting or opening body for unloading material.",
    "Haul Truck: A large, heavy truck for mining to transport bulk materials.",
    "Scraper or Tractor: A vehicle for moving, digging, and leveling soil.",
    "Front Loader or Bulldozer: A machine for scooping earth or pushing material.",
    "Excavator: A machine for removing soil or other material.",
    "Cement Mixer: A vehicle equipped for mixing cement.",
    "Ground Grader: A machine with a long blade for creating a flat surface.",
    "Hut or Tent: Simple shelters made from local materials or fabric.",
    "Shed: A simple structure typically for storage or workspaces.",
    "Building: A structure with a roof and walls.",
    "Aircraft Hangar: A building for storing and maintaining aircraft.",
    "Damaged Building: A building that has suffered structural damage.",
    "Facility: Buildings or places providing a particular service.",
    "Construction Site: A place where a building or infrastructure is being built.",
    "Vehicle Lot: A space for storing or selling vehicles.",
    "Helipad: A landing area for helicopters.",
    "Storage Tank: A large container for storing liquid or gas.",
    "Shipping Container Lot: A place where shipping containers are stored.",
    "Shipping Container: A standardized metal box for transport and storage of goods.",
    "Pylon: A tall structure supporting overhead power lines or for communication.",
    "Tower: A tall, narrow building or structure for support, observation, communication, etc."
    'Windmill: Multi-blade shape, often in white or silver color.'
]

categories_xview_visual_descriptions = [    "An elongated structure with two lateral extensions on each side, typically viewed in a tarmac or runway setting.",    "Similar to fixed-wing aircraft but visibly smaller in size, possibly with less complex shapes and details.",    "Large, elongated structures with wings, often located near terminal buildings or on runways. Passenger planes might have visible windows, while cargo planes may appear bulkier.",    "Smaller than aircraft, characterized by a main circular rotor on top and a smaller tail rotor.",    "Small, frequently rectangular or elliptical shapes, often found on roads or parking lots.",    "Smaller versions of passenger vehicles, typically with a compact, oval shape.",    "Long rectangular shapes, larger than cars, and usually seen on roads or in bus depots.",    "Similar to small cars but with a distinctly separated cab and cargo area.",    "Similar to pickup trucks, possibly with specialized equipment or attachments.",    "Large, rectangular shapes often found on highways or in industrial areas.",    "Large vehicles with an extended rectangular shape, potentially carrying goods.",    "A larger version of a truck, coupled with a big rectangular 'box' behind it.",    "Similar to trucks but without any attached trailer.",    "Detached elongated rectangular structures often seen in parking areas or on the road behind vehicles.",    "A truck tractor attached to a flat, long, rectangular structure.",    "A truck tractor with a large, circular, tank-like structure attached behind.",    "Trucks with long, crane-like structures attached.",    "Typically elongated, linear shapes on railway tracks.",    "Rectangular structures on railway tracks, often attached to other cars forming a train.",    "Similar to passenger cars but possibly boxier, as they're designed to carry goods.",    "Similar to cargo cars, but the upper structure might appear flat or empty.",    "Elongated structures on railway tracks, with a cylindrical shape indicative of liquid storage.",    "The front part of a train, usually bulkier and larger than the attached cars.",    "Varying sizes and shapes located in bodies of water, can range from small to large.",    "Small, often elongated shapes in water bodies.",    "Similar to motorboats but may display a triangular shape indicating the presence of sails.",    "Small, robust watercraft often located near larger ships or docks.",    "Long, flat structures in water, used for cargo transport.",    "Boats of various sizes, usually with additional structures for nets or equipment.",    "Large, rectangular structures often found near ports or in open water.",    "Large, luxurious boats often with a distinct streamlined shape.",    "Huge, elongated maritime vessels with a structure that appears 'stacked' due to the presence of containers.",    "Large, elongated maritime vessels with a bulbous front and rear section.",    "Vehicles of various shapes and sizes, often with unique structural elements like booms, buckets, or blades.",    "Tall, thin structures with a long horizontal part at the top, often seen at construction sites.",    "Large, towering structures usually found at docks, characterized by a 'T' or 'U' shape.",    "Industrial vehicles with a distinct arm-like structure, often seen at ports or warehouses.",    "Large, boxy vehicles often located in ports, with space underneath to transport containers.",    "Vehicles with a large, extendable boom arm.",    "Distinct vehicles with a large, often raised rear cargo section.",    "Large trucks with enormous rear section, often found in mining or quarry sites.",    "Often seen as small-to-medium sized vehicles with attached equipment for moving or grading soil.",    "Vehicles with a large, scoop-like front attachment.",    "Machines with a long, boom arm and bucket, used for digging.",    "Trucks with a large, rotating drum-like structure at the back.",    "Construction vehicles with a long, flat blade between the front and rear wheels.",    "Small, often round or square structures, less rigid in appearance than buildings.",    "Small, standalone structures often found in residential or rural areas.",    "Large, square or rectangular structures, usually with a consistent pattern (windows, balconies).",    "Large, usually rectangular buildings found near airfields, often with large doors.",    "Buildings with irregular, broken outlines or piles of debris around.",    "Complex of multiple buildings and structures, often found in industrial or military areas.",    "Areas with multiple types of vehicles, cranes, and partially completed structures.",    "Large open areas filled with multiple small to medium-sized objects (vehicles).",    "Flat, open surfaces, often circular or square, sometimes marked with a 'H'.",    "Large, often round structures, usually found in industrial areas.",    "Large, open spaces filled with multiple small, rectangular objects (containers).",    "Small, rectangular objects with uniform size, often seen in groups.",    "Tall, thin structures, often found in lines across landscapes.",    "Tall structures with a wider base and narrower top, often isolated."]


categories_dior = ['airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
               'golffield', 'harbor',
               'overpass', 'ship', 'stadium', 'storagetank',
               'tenniscourt', 'trainstation', 'vehicle', 'airport', 'basketballcourt', 'groundtrackfield', 'windmill']

categories_dior_visual_descriptions = [
    "Looks like a long object with side parts at a right angle. There is a big difference between the object and the area around it.",
    "Often shows a shape like a diamond with a big round or oval outer area. Different color parts could mean grass and areas inside the field.",
    "Appears as a straight line (or sometimes curved) over water or a low area, with a clear difference with the area below it.",
    "Can be seen as a tall, thin, upright shape, sometimes with smoke coming out of it, based on when the image was taken and if the factory is working.",
    "Appears as a long, thick line often across water, making a clear split between the water and the land.",
    "Often a mix of many shapes showing buildings, parking spots, and roads that join each other.",
    "Usually seen as a square or rectangle with many thin lines coming out of it, next to a highway.",
    "Big green areas broken by smaller sand parts and small round spots, often with smooth and curved edges.",
    "A big water area with many long shapes, often surrounded by different types of buildings like docks, storage buildings, and tall machines for lifting.",
    "Appears as a line or a curved shape over another road or train track, making shadows under it.",
    "Appears as a long shape in water, often standing out against the water surface.",
    "Usually round, oval, or rectangle with a clear inside area and outside shape. The inside often has shapes showing seating areas.",
    "Can be seen as big round shapes, often in groups, usually in industrial areas.",
    "Appears as rectangle areas, often in pairs, with clear lines showing the borders and the middle line.",
    "A complicated area with many lines side by side cut by other shapes and connected to nearby roads.",
    "Depending on how clear the image is, can be tiny rectangle or oval shapes. They are often found on roads or in parking areas.",
    "A big open area with many long shapes and a group of buildings. Planes and small objects might be seen.",
    "Appears as a rectangle with clear lines inside, including a circle in the center and two smaller circles at each end.",
    "Often oval, with inside lanes and different areas for different sports events.",
    "Can be seen as a small spot with long, thin lines coming out of it, often in groups in open areas."
]



model = SentenceTransformer('all-MiniLM-L6-v2')

#Compute embedding for both lists
embeddings1 = model.encode(categories_xview_visual_descriptions, convert_to_tensor=True)
print(embeddings1.shape)
embeddings = embeddings1 / np.linalg.norm(embeddings1, ord=2, axis=1, keepdims=True)
bg_embedding = np.random.rand(1, embeddings1.shape[1])
full_embeddings = np.concatenate([bg_embedding, embeddings1])
# print(full_embeddings.shape)
# np.save('xview_background_seen_and_unseen_word_vec_original_sentence_transformer.npy', full_embeddings)
# assuming `embeddings` is a 2D array of your embeddings, shape (21, 1024)
similarity_matrix = cosine_similarity(embeddings)
from scipy.special import softmax
similarity_matrix_normalized = softmax((similarity_matrix-np.eye(embeddings.size(0)))/0.1, axis=1)
similarity_matrix_normalized = softmax((similarity_matrix-np.eye(embeddings.size(0)))/0.05, axis=1)
np.save('sim_matrix_xview.npy', expand_similarity_matrix(similarity_matrix_normalized))
print(similarity_matrix_normalized)
distance_matrix = 1 - cosine_similarity(embeddings)

linkage_matrix = linkage(distance_matrix, method='average')


# Plot the dendrogram
plt.figure(figsize=(15, 7))
# plt.subplot(1, 2, 1)
dendrogram(linkage_matrix, labels=categories_xview)
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.3)

# Plot the cosine similarity matrix
# plt.subplot(1, 2, 2)
# plt.imshow(similarity_matrix, cmap='hot', vmin=0, vmax=1, aspect='auto')
# plt.colorbar(label='Cosine Similarity')
# plt.xticks(range(len(categories_xview)), categories_xview, rotation='vertical')
# plt.yticks(range(len(categories_xview)), categories_xview)

plt.tight_layout()
plt.show()

# plt.figure(figsize=(15, 7))
# plt.imshow(similarity_matrix_normalized, cmap='hot', vmin=0, vmax=1, aspect='auto')
# plt.colorbar(label='Cosine Similarity')
# plt.xticks(range(len(categories_xview)), categories_dior, rotation='vertical')
# plt.yticks(range(len(categories_xview)), categories_dior)

# plt.tight_layout()
# plt.show()


# categories_description = \
# ['Airplane: Streamlined body with two wings on each side.',
# 'Baseball field: Half-circle or oval shape surrounded by grass, typically yellow in color, with stands around it.',
# 'Bridge: Generally straight or curved structure, usually gray or concrete in color.',
# 'Chimney: Round cylindrical structure, typically white or gray in color.',
# 'Dam: Long and narrow shape with embankments on both sides, often in concrete gray, surrounded by lakes or rivers.',
# 'Expressway Service area: Rectangular or irregular shape surrounded by parking spaces, buildings, and greenery.',
# 'Expressway toll station: Rectangular shape with lanes and toll booths.',
# 'Golf field: Expansive area of green grass, often surrounded by lakes or trees.',
# 'Harbor: Body of water with a curved shape.',
# 'Overpass: Intersection of several roads with a bridge-like structure.',
# 'Ship: Various shapes and sizes, commonly seen in water.',
# 'Stadium: Circular or oval shape surrounded by stands or seating.',
# 'Storage tank: Typically a cylindrical shape, often silver or white in color.',
# 'Tennis court: Rectangular shape, typically with a green playing surface.',
# 'Train station: Rectangular or complex shape with platforms, tracks, and buildings.',
# 'Vehicle: Various types and colors.',
# 'Airport: Large facility, typically with a rectangular or irregular outline.',
# 'Basketball court: Rectangular shape with basketball hoops and markings on the ground.',
# 'Ground track field: Rectangular or square shape, often in red clay, grass, or hard court, possibly with surrounding stands.',
# 'Windmill: Multi-blade shape, often in white or silver color.'
# ]

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
    new_sim_mat[0, 0] = 0

    return new_sim_mat