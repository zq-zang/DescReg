from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

categories = ["fixed wing aircraft", "small aircraft", "passenger plane or cargo plane", "helicopter", "passenger vehicle", "small car", "bus",
                "pickup truck", "utility truck", "truck", "cargo truck", "truck tractor with box trailer", "truck tractor", "trailer", "truck tractor with flatbed trailer",
                "truck tractor with liquid tank", "crane truck", "railway vehicle", "passenger car", "cargo car or container car", "flat car", "tank car", "locomotive",
                "maritime vessel", "motorboat", "sailboat", "tugboat", "barge", "fishing vessel", "ferry", "yacht", "container ship", "oil tanker", "engineering vehicle",
                "tower crane", "container crane", "reach stacker", "straddle carrier", "mobile crane", "dump truck", "haul truck", "scraper or tractor", "front loader or bulldozer",
                "excavator", "cement mixer", "ground grader", "hut or tent", "shed", "building", "aircraft hangar", "damaged building", "facility", "construction site", "vehicle lot",
                "helipad", "storage tank", "shipping container lot", "shipping container", "pylon", "tower"]

categories_semantic_descriptions = \
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

categories_visual_descriptions = [    "An elongated structure with two lateral extensions on each side, typically viewed in a tarmac or runway setting.",
                                      "Similar to fixed-wing aircraft but visibly smaller in size, possibly with less complex shapes and details.",
                                      "Large, elongated structures with wings, often located near terminal buildings or on runways. Passenger planes might have visible windows, while cargo planes may appear bulkier.",
                                      "Smaller than aircraft, characterized by a main circular rotor on top and a smaller tail rotor.",
                                      "Small, frequently rectangular or elliptical shapes, often found on roads or parking lots.",
                                      "Smaller versions of passenger vehicles, typically with a compact, oval shape.",
                                      "Long rectangular shapes, larger than cars, and usually seen on roads or in bus depots.",
                                      "Similar to small cars but with a distinctly separated cab and cargo area.",
                                      "Similar to pickup trucks, possibly with specialized equipment or attachments.",
                                      "Large, rectangular shapes often found on highways or in industrial areas.",
                                      "Large vehicles with an extended rectangular shape, potentially carrying goods.",
                                      "A larger version of a truck, coupled with a big rectangular 'box' behind it.",
                                      "Similar to trucks but without any attached trailer.",
                                      "Detached elongated rectangular structures often seen in parking areas or on the road behind vehicles.",
                                      "A truck tractor attached to a flat, long, rectangular structure.",
                                      "A truck tractor with a large, circular, tank-like structure attached behind.",
                                      "Trucks with long, crane-like structures attached.",
                                      "Typically elongated, linear shapes on railway tracks.",
                                      "Rectangular structures on railway tracks, often attached to other cars forming a train.",
                                      "Similar to passenger cars but possibly boxier, as they're designed to carry goods.",
                                      "Similar to cargo cars, but the upper structure might appear flat or empty.",
                                      "Elongated structures on railway tracks, with a cylindrical shape indicative of liquid storage.",
                                      "The front part of a train, usually bulkier and larger than the attached cars.",
                                      "Varying sizes and shapes located in bodies of water, can range from small to large.",    "Small, often elongated shapes in water bodies.",    "Similar to motorboats but may display a triangular shape indicating the presence of sails.",    "Small, robust watercraft often located near larger ships or docks.",    "Long, flat structures in water, used for cargo transport.",    "Boats of various sizes, usually with additional structures for nets or equipment.",    "Large, rectangular structures often found near ports or in open water.",    "Large, luxurious boats often with a distinct streamlined shape.",    "Huge, elongated maritime vessels with a structure that appears 'stacked' due to the presence of containers.",    "Large, elongated maritime vessels with a bulbous front and rear section.",    "Vehicles of various shapes and sizes, often with unique structural elements like booms, buckets, or blades.",    "Tall, thin structures with a long horizontal part at the top, often seen at construction sites.",    "Large, towering structures usually found at docks, characterized by a 'T' or 'U' shape.",    "Industrial vehicles with a distinct arm-like structure, often seen at ports or warehouses.",    "Large, boxy vehicles often located in ports, with space underneath to transport containers.",    "Vehicles with a large, extendable boom arm.",    "Distinct vehicles with a large, often raised rear cargo section.",    "Large trucks with enormous rear section, often found in mining or quarry sites.",    "Often seen as small-to-medium sized vehicles with attached equipment for moving or grading soil.",    "Vehicles with a large, scoop-like front attachment.",    "Machines with a long, boom arm and bucket, used for digging.",    "Trucks with a large, rotating drum-like structure at the back.",    "Construction vehicles with a long, flat blade between the front and rear wheels.",    "Small, often round or square structures, less rigid in appearance than buildings.",    "Small, standalone structures often found in residential or rural areas.",    "Large, square or rectangular structures, usually with a consistent pattern (windows, balconies).",    "Large, usually rectangular buildings found near airfields, often with large doors.",    "Buildings with irregular, broken outlines or piles of debris around.",    "Complex of multiple buildings and structures, often found in industrial or military areas.",    "Areas with multiple types of vehicles, cranes, and partially completed structures.",    "Large open areas filled with multiple small to medium-sized objects (vehicles).",    "Flat, open surfaces, often circular or square, sometimes marked with a 'H'.",    "Large, often round structures, usually found in industrial areas.",    "Large, open spaces filled with multiple small, rectangular objects (containers).",    "Small, rectangular objects with uniform size, often seen in groups.",    "Tall, thin structures, often found in lines across landscapes.",    "Tall structures with a wider base and narrower top, often isolated."]

categories_visual_descriptions_nosemnatic = [
    "An elongated shape with two smaller extensions on each side.",
    "A compact, elliptical shape with smaller side extensions.",
    "A large structure with elongated central body and lateral wings, can display array of small dots (windows) or bulkier aspect.",
    "A compact structure featuring a circular detail on top and a smaller extension on one end.",
    "Small, often rectangular or elliptical shapes.",
    "Compact, often oval or slightly rectangular shapes.",
    "Longer rectangular shapes, typically larger than the aforementioned oval shapes.",
    "Rectangular shape with a distinctly separated front and rear.",
    "Similar to the separated-front-and-rear shape but with additional details or extensions.",
    "Large, often rectangular shapes.",
    "Extended rectangular shapes, may feature smaller details.",
    "A large rectangle attached to a smaller rectangular structure.",
    "A large rectangle with a cab-like structure at one end.",
    "Elongated detached rectangles, sometimes with additional structural details.",
    "A rectangle with a flat, elongated structure attached behind.",
    "A rectangle with a large, circular structure attached behind.",
    "Rectangular structure with a long extension attached.",
    "Elongated shapes located on parallel lines.",
    "Rectangular shapes on parallel lines, sometimes coupled together.",
    "Boxy shapes on parallel lines, sometimes featuring additional structural details.",
    "Shapes on parallel lines with flat or empty upper structure.",
    "Cylindrical shapes located on parallel lines.",
    "Larger, boxier shape at the front of a series of rectangles on parallel lines.",
    "Various sizes and shapes located in blue or green expanses (water).",
    "Small, often elongated shapes in blue or green expanses.",
    "Elongated shapes in water displaying a triangular detail.",
    "Compact, robust shapes often located near larger structures in water.",
    "Long, flat shapes in blue or green expanses.",
    "Various sizes of shapes in water, with additional structural details.",
    "Large, rectangular shapes often found in blue or green expanses.",
    "Large, streamlined shapes in water.",
    "Huge, elongated shapes in water that appear 'stacked'.",
    "Large, elongated shapes in water with a bulbous front and rear section.",
    "Shapes with unique structural elements like booms, buckets, or blades.",
    "Tall, thin shapes with a long horizontal detail at the top.",
    "Tall structures usually characterized by a 'T' or 'U' shape.",
    "Shapes featuring a distinct arm-like extension.",
    "Large, boxy shapes with a clear underpass.",
    "Shapes featuring a large, extendable arm.",
    "Shapes featuring a large, often raised rear section.",
    "Large shapes with a distinct rear section.",
    "Medium sized shapes with additional equipment details.",
    "Shapes featuring a large, scoop-like front detail.",
    "Shapes with a long, arm and bucket-like detail.",
    "Shapes with a large, circular structure at the rear.",
    "Shapes with a long, flat detail between front and rear.",
    "Small, often round or square shapes.",
    "Small standalone structures.",
    "Large, square or rectangular shapes with consistent patterning.",
    "Large, rectangular structures with very large entry points.",
    "Structures with irregular, broken outlines or surrounding smaller shapes (debris).",
    "Complex of multiple shapes and structures.",
    "Areas filled with multiple shapes, and partially completed structures.",
    "Large open areas filled with multiple small to medium-sized shapes.",
    "Flat, open surfaces, often circular or square.",
    "Large, often circular structures.",
    "Large, open spaces filled with multiple small, rectangular shapes.",
    "Small, rectangular shapes with uniform sizing.",
    "Tall, thin structures, often found in lines.",
    "Tall structures with a wider base and narrower top."
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
embeddings1 = model.encode(categories_visual_descriptions_nosemnatic, convert_to_tensor=True)
embeddings = embeddings1 / np.linalg.norm(embeddings1, ord=2, axis=1, keepdims=True)
bg_embedding = np.random.rand(1, embeddings1.shape[1])
full_embeddings = np.concatenate([bg_embedding, embeddings1])
# np.save('xview_background_seen_and_unseen_word_vec_original_sentence_transformer.npy', full_embeddings)

similarity_matrix = cosine_similarity(embeddings)
sim_matrix_xview_cosine = expand_similarity_matrix(similarity_matrix)
np.save('sim_matrix_xview_cosine.npy', sim_matrix_xview_cosine)
similarity_matrix_normalized = softmax((similarity_matrix-np.eye(embeddings.size(0)))/temperature, axis=1)
similarity_matrix_normalized += np.eye(embeddings.size(0))
# add background dimensioin
sim_matrix_xview_normalized = expand_similarity_matrix(similarity_matrix_normalized)
np.save('sim_matrix_xview_normalized_{}.npy'.format(temperature), sim_matrix_xview_normalized)
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
plt.imshow(sim_matrix_xview_normalized, cmap='hot', vmin=0, vmax=1, aspect='auto')
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(len(categories)), categories, rotation='vertical')
plt.yticks(range(len(categories)), categories)
plt.tight_layout()
plt.show()
