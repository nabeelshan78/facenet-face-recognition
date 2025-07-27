import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
from PIL import Image # Using PIL for image handling with Streamlit
import io # For handling BytesIO objects from uploaded files

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Face Recognition & Verification App",
    layout="centered", # Can be "wide" for more space
    initial_sidebar_state="expanded"
)

st.title("Face Recognition & Verification App 🧠🔍")
st.markdown("---")

# --- Model and Database Loading (Cached for Performance) ---

@st.cache_resource
def load_facenet_model():
    """
    Loads the FaceNet model architecture and weights.
    Uses st.cache_resource to load the model only once.
    """
    try:
        json_file_path = 'keras-facenet-h5/model.json'
        h5_file_path = 'keras-facenet-h5/model.h5'

        # Check if model files exist
        if not os.path.exists(json_file_path):
            st.error(f"Error: Model architecture file not found at `{json_file_path}`. Please ensure the `keras-facenet-h5` folder is in the same directory.")
            st.stop()
        if not os.path.exists(h5_file_path):
            st.error(f"Error: Model weights file not found at `{h5_file_path}`. Please ensure the `keras-facenet-h5` folder is in the same directory.")
            st.stop()

        with open(json_file_path, 'r') as json_file:
            loaded_json = json_file.read()
        
        model = model_from_json(loaded_json)
        model.load_weights(h5_file_path)
        return model
    except Exception as e:
        st.error(f"Failed to load FaceNet model: {e}")
        st.stop()

# Define initial people and their image paths
INITIAL_PEOPLE = [
    ("danielle", "images/danielle.png"),
    ("younes", "images/younes.jpg"),
    ("tian", "images/tian.jpg"),
    ("andrew", "images/andrew.jpg"),
    ("kian", "images/kian.jpg"),
    ("dan", "images/dan.jpg"),
    ("sebastiano", "images/sebastiano.jpg"),
    ("bertrand", "images/bertrand.jpg"),
    ("kevin", "images/kevin.jpg"),
    ("felix", "images/felix.jpg"),
    ("benoit", "images/benoit.jpg"),
    ("arnaud", "images/arnaud.jpg")
]

@st.cache_data(show_spinner="Pre-calculating database embeddings...")
def load_initial_database_embeddings(_model, people_list):
    """
    Loads images from the 'images' directory and computes their FaceNet embeddings.
    Uses st.cache_data to compute embeddings only once for the initial set.
    Returns both the database dictionary and the list of people with paths.
    """
    database = {}
    people_data = [] # To store (name, path) for display and future use

    # Check if the images directory exists
    if not os.path.exists('images'):
        st.error("Error: 'images' directory not found. Please ensure it exists and contains the database images.")
        st.stop()

    for name, path in people_list:
        if not os.path.exists(path):
            st.warning(f"Warning: Image for **{name.capitalize()}** not found at `{path}`. Skipping this entry.")
            continue
        try:
            database[name] = img_to_encoding(path, _model)
            people_data.append((name, path))
        except Exception as e:
            st.warning(f"Could not process image for **{name.capitalize()}** at `{path}`: {e}")
    return database, people_data

# --- FaceNet Helper Functions ---

# Helper function to get image source (path string or UploadedFile object) for a given person
def get_image_source_for_person(name, initial_people_list, dynamic_people_list):
    """
    Retrieves the image source (path string or UploadedFile object) for a given person name.
    """
    # Check initial people first
    for n, p_source in initial_people_list:
        if n == name:
            return p_source # This will be a string path

    # Check dynamically added people
    for n, p_source in dynamic_people_list:
        if n == name:
            return p_source # This will be an UploadedFile object
    return None

def img_to_encoding(img_source, model):
    """
    Converts an image (path string or BytesIO buffer from st.file_uploader)
    to its FaceNet embedding.
    """
    if isinstance(img_source, str): # It's a file path
        img = tf.keras.preprocessing.image.load_img(img_source, target_size=(160, 160))
    elif isinstance(img_source, (io.BytesIO, st.runtime.uploaded_file_manager.UploadedFile)): # It's an uploaded file buffer or UploadedFile object
        # If it's an UploadedFile, get its BytesIO content
        if isinstance(img_source, st.runtime.uploaded_file_manager.UploadedFile):
            img_source.seek(0) # Reset pointer to the beginning of the file
            img_stream = io.BytesIO(img_source.getvalue())
        else: # Already a BytesIO
            img_stream = img_source
        
        img = Image.open(img_stream)
        img = img.resize((160, 160), Image.LANCZOS) # Use LANCZOS for high-quality downsampling
    else:
        raise ValueError("img_source must be a file path (str), BytesIO object, or UploadedFile object.")
    
    img = np.array(img).astype('float32')
    img = np.around(img / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

def verify(img_source, identity, database, model, threshold=0.7):
    """
    Verifies if the person in the given image is who they claim to be.
    Returns distance and a boolean indicating if the door should be open.
    """
    try:
        img_encoding = img_to_encoding(img_source, model)
    except Exception as e:
        st.error(f"Error processing test image for verification: {e}")
        return None, False

    if identity not in database:
        return None, False # Identity not in database
    
    db_encoding = database[identity]
    dist = np.linalg.norm(img_encoding - db_encoding)
    
    door_open = dist < threshold
    
    return dist, door_open

def who_is_it(img_source, database, model, threshold=0.7):
    """
    Identifies the person in the given image from the database.
    Returns the minimum distance and the identified identity.
    """
    try:
        img_encoding = img_to_encoding(img_source, model)
    except Exception as e:
        st.error(f"Error processing test image for recognition: {e}")
        return None, "Error"

    min_dist = float('inf')
    identity = "Unknown"
    
    if not database:
        return None, "No faces in database"

    for (name, db_enc) in database.items():
        dist = np.linalg.norm(img_encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
            
    if min_dist > threshold:
        identity = "Not in database"
        
    return min_dist, identity

# --- Initialize Session State for Dynamic Database ---
if 'dynamic_database' not in st.session_state:
    st.session_state.dynamic_database = {}
if 'dynamic_people' not in st.session_state:
    st.session_state.dynamic_people = []

# --- Load Model and Initial Database ---
model = load_facenet_model()
initial_database, initial_people_data = load_initial_database_embeddings(model, INITIAL_PEOPLE)

# Merge initial and dynamic database entries
current_database = {**initial_database, **st.session_state.dynamic_database}
# current_people_list is used for dropdowns and display, so it needs both initial paths and dynamic UploadedFile objects
current_people_list = initial_people_data + st.session_state.dynamic_people

# Filter out people with missing images from the dropdown
available_people_names = [name for name, _ in current_people_list if name in current_database]

# --- Sidebar Navigation ---
st.sidebar.header("Navigation 🧭")
task_choice = st.sidebar.radio(
    "Choose an operation:",
    ("🔍 Face Verification", "🧠 Face Recognition", "➕ Manage Database", "💡 How it Works")
)

st.sidebar.markdown("---")
st.sidebar.info("This robust application has been meticulously developed by me, a passionate and aspiring" \
" AI/ML Engineer and a third-year undergraduate in Software Engineering at NUST Islamabad. This project " \
"showcases my dedication to building high-impact solutions and my commitment to advancing my skills in " \
"in AI/ML. I am excited to continue " \
"contributing to the field through practical applications and research.")


# --- Helper for Image Input ---
def get_image_input(key_prefix):
    """
    Provides options to upload a new image or select from database.
    Returns the image source (BytesIO or path) and a display name.
    """
    input_method = st.radio(
        f"Choose image source for {key_prefix}:",
        ("Upload a new image", "Select from database"),
        key=f"{key_prefix}_input_method"
    )

    image_source = None
    display_image = None
    display_name = ""

    if input_method == "Upload a new image":
        uploaded_file = st.file_uploader(f"Upload an image for {key_prefix}:", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_uploader")
        if uploaded_file:
            image_source = io.BytesIO(uploaded_file.getvalue()) # Always convert to BytesIO for consistency
            display_image = uploaded_file
            display_name = "Uploaded Image"
    else: # Select from database
        # Prepend a placeholder to the options
        db_options = ["--- Select a person from database ---"] + [name.capitalize() for name in available_people_names]
        
        selected_db_person_display = st.selectbox(
            f"Select a person from the database for {key_prefix}:",
            options=db_options,
            key=f"{key_prefix}_db_selector"
        )
        
        # Only proceed if a valid person is selected (not the placeholder)
        if selected_db_person_display and selected_db_person_display != "--- Select a person from database ---":
            selected_db_person = selected_db_person_display.lower() # Convert back to lowercase for lookup
            
            # Get the actual image source (path string or UploadedFile object)
            image_source_raw = get_image_source_for_person(selected_db_person, initial_people_data, st.session_state.dynamic_people)

            if image_source_raw:
                image_source = image_source_raw # This can be a path or an UploadedFile object
                
                # For display, open the image correctly based on its type
                if isinstance(image_source_raw, str):
                    if os.path.exists(image_source_raw):
                        display_image = Image.open(image_source_raw)
                        display_name = f"Database Image ({selected_db_person.capitalize()})"
                    else:
                        st.warning(f"Image file for {selected_db_person.capitalize()} not found on disk.")
                elif isinstance(image_source_raw, st.runtime.uploaded_file_manager.UploadedFile):
                    image_source_raw.seek(0) # Reset pointer for display
                    display_image = Image.open(io.BytesIO(image_source_raw.getvalue()))
                    display_name = f"Database Image (Dynamic: {selected_db_person.capitalize()})"
                else:
                    st.error("Unexpected image source type for display.")
            else:
                st.warning(f"Image data for {selected_db_person.capitalize()} not found in database.")
        else:
            st.info("Please select a person from the database.")

    return image_source, display_image, display_name


# --- Face Verification Section ---
if task_choice == "🔍 Face Verification":
    st.header("Face Verification 🔐")
    st.write("Verify if an image matches a **claimed identity** from the database.")

    st.subheader("1. Claimed Identity")
    if not available_people_names:
        st.warning("No people in the database. Please add some via 'Manage Database'.")
        selected_identity = None
    else:
        # Prepend a placeholder to the options
        identity_options = ["--- Select a claimed identity ---"] + [name.capitalize() for name in available_people_names]
        
        selected_identity_display = st.selectbox(
            "Who are you claiming this person to be? (Select from database)",
            options=identity_options,
            key="verify_identity_selector"
        )
        
        # Convert back to lowercase for lookup if a valid option is selected
        if selected_identity_display and selected_identity_display != "--- Select a claimed identity ---":
            selected_identity = selected_identity_display.lower()
        else:
            selected_identity = None # No valid identity selected yet
            st.info("Please select a claimed identity.")
    
    identity_display_image = None
    if selected_identity: # Only try to display if a valid identity is selected
        # Get the raw source (path or UploadedFile)
        identity_image_source_raw = get_image_source_for_person(selected_identity, initial_people_data, st.session_state.dynamic_people)
        
        if identity_image_source_raw:
            if isinstance(identity_image_source_raw, str):
                if os.path.exists(identity_image_source_raw):
                    identity_display_image = Image.open(identity_image_source_raw)
                else:
                    st.warning(f"Database image for {selected_identity.capitalize()} not found on disk.")
            elif isinstance(identity_image_source_raw, st.runtime.uploaded_file_manager.UploadedFile):
                identity_image_source_raw.seek(0) # Reset pointer for display
                identity_display_image = Image.open(io.BytesIO(identity_image_source_raw.getvalue()))
            else:
                st.error("Unexpected identity image source type for display.")
        
        if identity_display_image:
            st.image(identity_display_image, caption=f"Database image of {selected_identity.capitalize()} (Claimed Identity)", width=150)
        st.info(f"Claimed Identity: **{selected_identity.capitalize()}**")

    st.subheader("2. Image to Verify")
    test_image_source, test_display_image, test_display_name = get_image_input("verification_test")

    st.markdown("---")

    # Only proceed with verification if both a test image and a claimed identity are selected
    if test_image_source and selected_identity:
        st.subheader("Verification Results 📊")
        col_test_img, col_identity_img, col_results = st.columns([1, 1, 2])

        with col_test_img:
            if test_display_image:
                st.image(test_display_image, caption=test_display_name, use_column_width=True)
            else:
                st.write("No test image to display.")
        
        with col_identity_img:
            if identity_display_image: # Use the already prepared display image
                st.image(identity_display_image, caption=f"Database Image ({selected_identity.capitalize()})", use_column_width=True)
            else:
                st.write("No identity image to display.")

        with col_results:
            with st.spinner(f"Verifying if the image matches {selected_identity.capitalize()}..."):
                distance, door_open = verify(test_image_source, selected_identity, current_database, model)
            
            if distance is None:
                st.error("❗ Could not perform verification. Check image and identity selection.")
            else:
                st.metric(label="Distance", value=f"{distance:.3f}")
                if door_open:
                    st.success(f"✅ **Welcome in! The image matches {selected_identity.capitalize()}.**")
                    st.balloons()
                else:
                    st.error(f"❌ **Go away! The image does NOT match {selected_identity.capitalize()}.**")
                st.info(f"Threshold for verification: 0.7")
    else:
        st.info("Please provide an image to verify and select a claimed identity to see results.")


# --- Face Recognition Section ---
elif task_choice == "🧠 Face Recognition":
    st.header("Face Recognition 🔎")
    st.write("Identify the person in an uploaded image from the database.")

    st.subheader("1. Provide Test Image")
    test_image_source, test_display_image, test_display_name = get_image_input("recognition_test")

    st.markdown("---")

    if test_image_source:
        st.subheader("Recognition Results 📊")
        col_rec_img, col_rec_results = st.columns([1, 2])

        with col_rec_img:
            if test_display_image:
                st.image(test_display_image, caption=test_display_name, use_column_width=True)
            else:
                st.write("No test image to display.")

        with col_rec_results:
            with st.spinner("Recognizing face..."):
                distance, identity = who_is_it(test_image_source, current_database, model)

            if distance is None:
                st.error("❗ Could not perform recognition. Check image selection.")
            elif identity == "No faces in database":
                st.warning("Database is empty. Please add some faces first.")
            else:
                st.metric(label="Predicted Identity", value=identity.capitalize())
                st.metric(label="Distance to Match", value=f"{distance:.3f}")

                if identity == "Not in database":
                    st.warning("🚫 **Person not found in the database.**")
                else:
                    st.success(f"🎉 Identified as: **{identity.capitalize()}**")
                    # Display the database image of the recognized person
                    recognized_image_source_raw = get_image_source_for_person(identity, initial_people_data, st.session_state.dynamic_people)
                    recognized_display_image = None
                    if recognized_image_source_raw:
                        if isinstance(recognized_image_source_raw, str):
                            if os.path.exists(recognized_image_source_raw):
                                recognized_display_image = Image.open(recognized_image_source_raw)
                        elif isinstance(recognized_image_source_raw, st.runtime.uploaded_file_manager.UploadedFile):
                            recognized_image_source_raw.seek(0) # Reset pointer for display
                            recognized_display_image = Image.open(io.BytesIO(recognized_image_source_raw.getvalue()))
                    
                    if recognized_display_image:
                        st.image(recognized_display_image, caption=f"Database image of {identity.capitalize()}", width=150)
                    else:
                        st.info(f"Database image for {identity.capitalize()} not available.")
                st.info(f"Threshold for recognition: 0.7")
    else:
        st.info("Please provide an image to perform recognition.")


# --- Manage Database Section ---
elif task_choice == "➕ Manage Database":
    st.header("Manage Face Database 🗃️")
    st.write("Add new persons to your database or view existing entries.")

    st.subheader("Add New Person")
    with st.expander("Click to add a new face to the database"):
        new_person_name = st.text_input("Enter new person's name (e.g., 'alice')", key="new_person_name").strip().lower()
        new_person_image = st.file_uploader("Upload image for new person:", type=["jpg", "jpeg", "png"], key="new_person_image_uploader")

        if st.button("Add Person to Database", key="add_person_button"):
            if not new_person_name:
                st.warning("Please enter a name for the new person.")
            elif not new_person_image:
                st.warning("Please upload an image for the new person.")
            elif new_person_name in current_database:
                st.warning(f"**{new_person_name.capitalize()}** already exists in the database. Please choose a different name.")
            else:
                try:
                    # Compute embedding for the new image
                    # Pass the UploadedFile object directly to img_to_encoding
                    new_embedding = img_to_encoding(new_person_image, model)
                    
                    # Store in session state
                    st.session_state.dynamic_database[new_person_name] = new_embedding
                    # Store the UploadedFile object directly in dynamic_people
                    st.session_state.dynamic_people.append((new_person_name, new_person_image)) 
                    
                    st.success(f"🎉 **{new_person_name.capitalize()}** added to database successfully!")
                    st.experimental_rerun() # Rerun to update dropdowns and database view
                except Exception as e:
                    st.error(f"Failed to add person: {e}")

    st.subheader("Current Database Entries")
    if current_people_list:
        st.write("Here are all the faces currently in your database:")
        
        # Display all entries in a grid
        cols_per_row = 3 # Adjust as needed for layout
        cols = st.columns(cols_per_row)
        
        for i, (name, image_source_raw) in enumerate(current_people_list):
            with cols[i % cols_per_row]:
                image_to_display = None
                try:
                    if isinstance(image_source_raw, str): # Initial images (path)
                        if os.path.exists(image_source_raw):
                            image_to_display = Image.open(image_source_raw)
                    elif isinstance(image_source_raw, st.runtime.uploaded_file_manager.UploadedFile): # Dynamic images (UploadedFile)
                        image_source_raw.seek(0) # Reset pointer for display
                        image_to_display = Image.open(io.BytesIO(image_source_raw.getvalue()))
                    
                    if image_to_display:
                        st.image(image_to_display, caption=name.capitalize(), width=100) # Smaller width for grid
                    else:
                        st.caption(f"Image for {name.capitalize()} not found.")
                except Exception as e:
                    st.caption(f"Error loading image for {name.capitalize()}: {e}")
    else:
        st.info("No persons currently in the database.")

# st.markdown("---")
# st.markdown("Feel free to experiment with different images and functionalities!")

# --- How it Works Section ---
elif task_choice == "💡 How it Works":
    st.header("How Face Recognition & Verification Works 💡")
    st.write("Let's break down the magic behind this application. At its core, we're using a powerful Deep Learning model called **FaceNet**.")
    st.markdown("---")

    st.subheader("1. Face Embeddings: The Core Idea")
    st.write(
        """
        Imagine you want to compare two faces to see if they belong to the same person. How do you do that mathematically?
        FaceNet solves this by transforming each face into a **"face embedding"**.
        """
    )
    st.markdown(
        """
        * **What is an Embedding?** It's a list of numbers (a vector) that represents the unique features of a face in a high-dimensional space. Think of it like a unique fingerprint for a face, but in numerical form.
        * **How FaceNet Creates Them:** FaceNet is a Convolutional Neural Network (CNN) trained to generate these embeddings. It learns to map images of the same person's face to very similar embeddings, and images of different people's faces to very different embeddings.
        * **Example:** If you have two images of 'Younes', FaceNet will produce two embedding vectors, say $f(Younes_1)$ and $f(Younes_2)$, that are very close to each other. If you have an image of 'Younes' and 'Danielle', their embeddings $f(Younes)$ and $f(Danielle)$ will be far apart.
        """
    )
    st.markdown("---")

    st.subheader("2. Measuring Similarity: Euclidean Distance")
    st.write(
        """
        Once we have these numerical embeddings, comparing faces becomes a simple math problem: calculating the distance between their embedding vectors.
        """
    )
    st.markdown(
        """
        * We use **Euclidean Distance** to measure how "far apart" two embeddings are in this high-dimensional space.
        * **Formula:** For two embeddings $f_1 = (x_1, y_1, ..., z_1)$ and $f_2 = (x_2, y_2, ..., z_2)$, the Euclidean distance $d$ is:
            $$ d(f_1, f_2) = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 + \dots + (z_1-z_2)^2} $$
        * **Interpretation:**
            * **Small Distance:** Indicates the faces are very similar, likely belonging to the same person.
            * **Large Distance:** Indicates the faces are very different, likely belonging to different people.
        """
    )
    st.markdown("---")

    st.subheader("3. Face Verification: 'Is this X?'")
    st.write(
        """
        This task answers the question: "Is the person in this image **X** (a specific claimed identity)?"
        """
    )
    st.markdown(
        """
        * **Process:**
            1.  Get the embedding of the **test image** ($f_{test}$).
            2.  Retrieve the stored embedding of the **claimed identity** from the database ($f_{claimed}$).
            3.  Calculate the **distance** between $f_{test}$ and $f_{claimed}$.
            4.  Compare this distance to a predefined **threshold** (e.g., 0.7 in this app).
        * **Result:**
            * If $d(f_{test}, f_{claimed}) < \text{threshold}$: ✅ "Welcome in!" (Match)
            * If $d(f_{test}, f_{claimed}) \ge \text{threshold}$: ❌ "Go away!" (No Match)
        """
    )
    st.markdown("---")

    st.subheader("4. Face Recognition: 'Who is this?'")
    st.write(
        """
        This task answers the question: "Who is the person in this image among all known individuals?"
        """
    )
    st.markdown(
        """
        * **Process:**
            1.  Get the embedding of the **test image** ($f_{test}$).
            2.  Iterate through **every single face embedding** ($f_{db}$) in your database.
            3.  Calculate the **distance** between $f_{test}$ and *each* $f_{db}$ in the database.
            4.  Find the database face that has the **minimum distance** to $f_{test}$.
            5.  Compare this *minimum distance* to the same predefined **threshold**.
        * **Result:**
            * If $\text{min\_distance} < \text{threshold}$: 🎉 "Identified as [Name]!"
            * If $\text{min\_distance} \ge \text{threshold}$: 🚫 "Not in the database."
        """
    )
    st.markdown("---")

    st.subheader("In Summary")
    st.write(
        """
        This application uses FaceNet to convert faces into numerical embeddings. By calculating the distance between these embeddings, it can determine how similar two faces are, enabling both verification (one-to-one comparison) and recognition (one-to-many comparison). The `0.7` threshold is a crucial parameter that determines the strictness of the match.
        """
    )
    st.markdown("---")
    st.info("For deeper dives, explore topics like triplet loss function (which FaceNet uses for training) and different CNN architectures!")

