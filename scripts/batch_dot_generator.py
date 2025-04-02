# %% Imports
import os, cv2, numpy as np, matplotlib.pyplot as plt, torch, json, csv, argparse
from matplotlib.widgets import Slider
from segment_anything import sam_model_registry, SamPredictor
print("Imports done.")

parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Force reprocessing of images')
args = parser.parse_args()
# %% Config
curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)

original_dir = '../stimuli/original'
gray_dir = '../stimuli/gray'
mooney_dir = '../stimuli/mooney'
output_dir = '../stimuli/output_dots'
master_csv = os.path.join(output_dir, 'master_log.csv')
sam_checkpoint = '../models/sam_vit_h_4b8939.pth'
model_type = "vit_h"

dot_color = (0, 0, 255)
dot_radius = 15
num_dots = 6  # means 6 for mooney, 6 for gray
sampling_points = 50
circle_radius_ratio = 0.5

os.makedirs(output_dir, exist_ok=True)

# get name of images in gray_dir
image_files = sorted([f for f in os.listdir(gray_dir) if f.endswith('.jpg')])

# remove .jpg from image_files
image_files = [os.path.splitext(f)[0] for f in image_files]
# save image_files to a text file
with open(os.path.join(output_dir, 'image_files.txt'), 'w') as f:
    for item in image_files:
        f.write("%s\n" % item)
        
        

# %% Setup SAM
print("Loading SAM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)
print("model loaded.")

# %% Functions
def get_center(image): h, w = image.shape[:2]; return w // 2, h // 2

def generate_circle_positions(center, radius, total_points):
    angles = np.linspace(0, 2 * np.pi, total_points, endpoint=False)
    return [(int(center[0] + radius * np.cos(a)), int(center[1] + radius * np.sin(a))) for a in angles]

def classify_positions(positions, mask):
    on, off = [], []
    for x, y in positions:
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            if mask[y, x] > 127: on.append((x, y))
            else: off.append((x, y))
    return on, off

def pick_evenly_spaced(points, num):
    if len(points) < num:
        raise ValueError("Not enough valid points.")
    step = len(points) // num
    return [points[i * step] for i in range(num)]

def filter_near_edges(points, image_shape, margin):
    h, w = image_shape[:2]
    return [(x, y) for x, y in points if margin <= x < (w - margin) and margin <= y < (h - margin)]

def filter_near_mask_edges(points, mask, min_distance):
    dist_transform = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
    return [(x, y) for x, y in points if dist_transform[y, x] > min_distance]

def filter_away_from_object(points, mask, min_distance):
    """
    Filters OFF dots to ensure they are at least `min_distance` away from the object (mask).
    """
    inverse_mask = (mask == 0).astype(np.uint8)
    dist_transform = cv2.distanceTransform(inverse_mask, cv2.DIST_L2, 5)
    return [(x, y) for x, y in points if dist_transform[y, x] > min_distance]


def draw_dots_and_circle(image, dot_positions, circle_positions, center, radius, dot_color, dot_radius):
    img_copy = image.copy()
    cv2.circle(img_copy, center, radius, (150, 150, 150), 2, cv2.LINE_AA)
    for x, y in dot_positions:
        cv2.circle(img_copy, (x, y), dot_radius, dot_color, -1)
    return img_copy

def save_dot_versions(base_img, positions, label, tag, base_name, save_folder, grayscale=False):
    dot_metadata = []
    for i, (x, y) in enumerate(positions, 1):
        if grayscale:
            out_img = cv2.cvtColor(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        else:
            out_img = base_img.copy()
        cv2.circle(out_img, (x, y), dot_radius, dot_color, thickness=-1)
        fname = f"{base_name}_{tag}_dot_{label}{i}.jpg"
        path = os.path.join(save_folder, fname)
        cv2.imwrite(path, out_img)
        dot_metadata.append({
            "image": fname,
            "label": label,
            "type": tag,
            "dot_number": i,
            "x": x,
            "y": y
        })
    return dot_metadata

# %% Init CSV
if not os.path.exists(master_csv):
    with open(master_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "type", "label", "dot_number", "x", "y"])
        writer.writeheader()

# %% Main loop
image_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.jpg')])
print(f"Found {len(image_files)} images.\n")

for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]
    output_subdir = os.path.join(output_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)
    metadata_path = os.path.join(output_subdir, 'metadata.json')

    if os.path.exists(metadata_path) and not args.force:
        print(f"Skipping {base_name} (already done)")
        continue

    print(f"\n🖼️ Processing: {base_name}")

    original_img = cv2.imread(os.path.join(original_dir, image_file))
    gray_img = cv2.imread(os.path.join(gray_dir, f"{base_name}_gray.jpg"))
    mooney_img = cv2.imread(os.path.join(mooney_dir, f"{base_name}_mooney.jpg"))

    if original_img is None or gray_img is None or mooney_img is None:
        print(f"⚠️ Missing images for {base_name}, skipping.")
        continue

    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    height, width = rgb_img.shape[:2]

    predictor.set_image(original_img)
    clicked_points = []

    def on_click(event):
        if event.xdata and event.ydata:
            clicked_points.append([int(event.xdata), int(event.ydata)])
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()

    print("Click object(s) to segment, then close window.")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb_img)
    ax.set_title(f"{base_name}: Click object(s)")
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if not clicked_points:
        print("⚠️ No points clicked. Skipping.")
        continue

    input_points = np.array(clicked_points)
    input_labels = np.ones(len(clicked_points))
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    # Save final mask
    mask_path = os.path.join(output_subdir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, combined_mask)

    # Create transparent RGBA masked object
    bgr = original_img.copy()
    alpha = (combined_mask > 0).astype(np.uint8) * 255  # 255 for object, 0 for background

    # Stack BGR + Alpha to RGBA
    rgba = cv2.merge((*cv2.split(bgr), alpha))

    object_path = os.path.join(output_subdir, f"{base_name}_object.png")
    cv2.imwrite(object_path, rgba)

    center = get_center(rgb_img)
    radius = int(min(height, width) * circle_radius_ratio)
    circle_positions = generate_circle_positions(center, radius, sampling_points)
    on_all, off_all = classify_positions(circle_positions, combined_mask)

    def filter_near_edges(points, image_shape, margin):
        h, w = image_shape[:2]
        return [(x, y) for x, y in points if margin <= x < (w - margin) and margin <= y < (h - margin)]

    def attempt_dot_pick(ratio):
        test_radius = int(min(height, width) * ratio)
        circle_pos = generate_circle_positions(center, test_radius, sampling_points)
        on_temp, off_temp = classify_positions(circle_pos, combined_mask)

        edge_margin = dot_radius + 25
        print(f"\n🔹 Circle radius = {test_radius}px")
        print(f"Original ON: {len(on_temp)}, OFF: {len(off_temp)}")
        
        on_temp = filter_near_mask_edges(on_temp, combined_mask, min_distance=edge_margin)
        off_temp = filter_away_from_object(off_temp, combined_mask, min_distance=edge_margin)
        circle_filtered = filter_near_edges(circle_pos, combined_mask.shape, edge_margin)
        
        print(f"After edge filter — ON: {len(on_temp)}, OFF: {len(off_temp)}")

        try:
            on_ext = pick_evenly_spaced(on_temp, num_dots * 2)
            off_ext = pick_evenly_spaced(off_temp, num_dots * 2)
            return on_ext, off_ext, circle_filtered, test_radius
        except:
            return None, None, circle_filtered, test_radius

    try:
        edge_margin = dot_radius + 25
        on_all = filter_near_mask_edges(on_all, combined_mask, min_distance=edge_margin)
        off_all = filter_away_from_object(off_all, combined_mask, min_distance=edge_margin)

        on_all_ext = pick_evenly_spaced(on_all, num_dots * 2)
        off_all_ext = pick_evenly_spaced(off_all, num_dots * 2)

    except ValueError:
        print("⚠️ Not enough dot positions. Use the slider to adjust radius.")
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.25)
        slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
        radius_slider = Slider(slider_ax, 'Radius %', 0.1, 0.9, valinit=circle_radius_ratio)
        img_display = ax.imshow(rgb_img)

        def update(val):
            on_ext, off_ext, circle_pos_filtered, updated_r = attempt_dot_pick(val)
            if on_ext and off_ext:
                preview = draw_dots_and_circle(rgb_img, on_ext[:num_dots], circle_pos_filtered, center, updated_r, dot_color, dot_radius)
                
                # 🔍 Optional: draw yellow dots to preview all valid filtered positions
                for x, y in circle_pos_filtered:
                    cv2.circle(preview, (x, y), 3, (0, 255, 255), -1)

                img_display.set_data(preview)
                fig.canvas.draw_idle()


        radius_slider.on_changed(update)
        update(circle_radius_ratio)
        plt.title("Adjust radius until valid dot layout appears. Then close.")
        plt.show()

        final_r = radius_slider.val
        on_all_ext, off_all_ext, circle_positions, radius = attempt_dot_pick(final_r)

        if not on_all_ext or not off_all_ext:
            print("❌ Still not enough points. Skipping.")
            continue

    # Divide into Mooney and Gray
    on_mooney = on_all_ext[::2]
    on_gray   = on_all_ext[1::2]
    off_mooney = off_all_ext[::2]
    off_gray   = off_all_ext[1::2]

    # Show preview
    circle_img = draw_dots_and_circle(rgb_img, [], circle_positions, center, radius, dot_color, dot_radius)
    mooney_on = draw_dots_and_circle(rgb_img, on_all_ext, circle_positions, center, radius, dot_color, dot_radius)
    mooney_off = draw_dots_and_circle(rgb_img, off_all_ext, circle_positions, center, radius, dot_color, dot_radius)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(circle_img); axs[0].set_title("Circle only")
    axs[1].imshow(mooney_off); axs[1].set_title("Dots OFF object")
    axs[2].imshow(mooney_on); axs[2].set_title("Dots ON object")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Save images and metadata
    meta = {
        "base_name": base_name,
        "click_points": clicked_points,
        "circle_radius": radius,
        "dots": []
    }

    meta["dots"] += save_dot_versions(mooney_img, on_mooney, "on", "mooney", base_name, output_subdir)
    meta["dots"] += save_dot_versions(mooney_img, off_mooney, "off", "mooney", base_name, output_subdir)
    meta["dots"] += save_dot_versions(gray_img, on_gray, "on", "gray", base_name, output_subdir, grayscale=True)
    meta["dots"] += save_dot_versions(gray_img, off_gray, "off", "gray", base_name, output_subdir, grayscale=True)

    with open(metadata_path, 'w') as f:
        json.dump(meta, f, indent=2)

    with open(master_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "type", "label", "dot_number", "x", "y"])
        for d in meta["dots"]:
            writer.writerow(d)

    print(f"✅ Done: {base_name}")
# %%
