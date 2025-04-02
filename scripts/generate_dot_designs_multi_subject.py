

import random
from itertools import cycle
import pandas as pd
import os

# === CONFIGURATION ===

VERBOSE_LEVEL = "summary"  # Options: "full", "summary", "none"
optimization_attemps = 100  # Number of attempts to optimize the design

stimulus_manipulations = ["regular", "upside_down", "horizontal_flip"]
dot_trials_per_image = 4
n_target_sets = 5 # this is basically the number of blocks - 1
images_per_set = 3 # numer of target images per block (set to a minimum of 3 to ensure at least 1 image per condition)
catch_ratio = 0.2  # set to a minimum of 0.2 to ensure at least 1 catch image across manipulations
blocks_total = n_target_sets + 1  # final block is dot-only

curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)


# === LOAD IMAGE LABELS ===
def load_usable_labels(filepath="image_files.txt"):
    with open(filepath, "r") as f:
        gray_images = [line.strip() for line in f if line.strip()]
    return [name.replace("_gray", "") for name in gray_images if name.endswith("_gray")]


# === CREATE TARGET & CATCH SETS ===
def create_image_sets(labels):
    random.shuffle(labels)
    n_target_images = n_target_sets * images_per_set
    n_catch_images = int(n_target_images * catch_ratio)

    total_needed = n_target_images + n_catch_images
    if len(labels) < total_needed:
        raise ValueError(f"Need at least {total_needed} labels, only {len(labels)} found.")

    target_pool = labels[:n_target_images]
    catch_pool = labels[n_target_images:n_target_images + n_catch_images]

    # Targets: A–E
    target_sets = {
        chr(65 + i): target_pool[i * images_per_set:(i + 1) * images_per_set]
        for i in range(n_target_sets)
    }

    # Distribute catch images randomly across block pairs (A–E)
    catch_sets = {chr(65 + i): [] for i in range(n_target_sets)}
    block_pairs = list(catch_sets.keys())  # block A–E
    random.shuffle(catch_pool)

    for i, catch_img in enumerate(catch_pool):
        pair_idx = i % len(block_pairs)
        catch_sets[block_pairs[pair_idx]].append(catch_img)

    return target_sets, catch_sets


# === ASSIGN MANIPULATIONS ===
def assign_manipulations(target_sets, catch_sets):
    manipulation_cycle = cycle(stimulus_manipulations)
    image_info = {}

    for sets, image_type in [(target_sets, "target"), (catch_sets, "catch")]:
        for img_list in sets.values():
            for img in img_list:
                image_info[img] = {
                    "manipulation": next(manipulation_cycle),
                    "image_type": image_type
                }
    return image_info


# === DOT TRIALS ===
def generate_dot_trials(images, phase_lookup, image_info, block):
    trials = []
    for img in images:
        for rep in range(dot_trials_per_image):
            dot_pos = "on" if rep % 2 == 0 else "off"
            dot_num = rep % 2 + 1
            trials.append({
                "block": block,
                "block_type": "dot",
                "image_id": img,
                "filename": f"{img}_mooney_dot_{dot_pos}{dot_num}",
                "image_type": image_info[img]["image_type"],
                "manipulation": image_info[img]["manipulation"],
                "phase": phase_lookup[img],
                "dot_position": dot_pos
            })
    return trials


# === RECOGNITION TRIALS ===
def generate_recognition_trials(pre_imgs, post_imgs, image_info, block):
    mooney_trials = []
    grayscale_trials = []

    for img in pre_imgs:
        mooney_trials.append({
            "block": block,
            "block_type": "recognition",
            "image_id": img,
            "filename": f"{img}_mooney",
            "image_type": image_info[img]["image_type"],
            "manipulation": image_info[img]["manipulation"],
            "phase": "pre",
            "dot_position": None
        })
        if image_info[img]["image_type"] == "target":
            grayscale_trials.append({
                "block": block,
                "block_type": "recognition",
                "image_id": img,
                "filename": f"{img}_gray",
                "image_type": image_info[img]["image_type"],
                "manipulation": image_info[img]["manipulation"],
                "phase": "gray",
                "dot_position": None
            })

    for img in post_imgs:
        mooney_trials.append({
            "block": block,
            "block_type": "recognition",
            "image_id": img,
            "filename": f"{img}_mooney",
            "image_type": image_info[img]["image_type"],
            "manipulation": image_info[img]["manipulation"],
            "phase": "post",
            "dot_position": None
        })

    random.shuffle(mooney_trials)
    return mooney_trials + grayscale_trials


# === SESSION BUILDER ===
def generate_full_session(filepath="image_files.txt", verbose=False):
    labels = load_usable_labels(filepath)
    target_sets, catch_sets = create_image_sets(labels)
    image_info = assign_manipulations(target_sets, catch_sets)

    all_trials = []
    previous_target_set = []
    previous_catch_set = []
    
    block_num = 0

    for block_idx in range(blocks_total):
        block_label = chr(65 + block_idx) if block_idx < n_target_sets else None

        target_pre = target_sets.get(block_label, [])
        catch_pre = catch_sets.get(block_label, [])

        target_post = previous_target_set
        catch_post = previous_catch_set

        pre_imgs = target_pre + catch_pre
        post_imgs = target_post + catch_post

        phase_lookup = {img: "pre" for img in pre_imgs}
        phase_lookup.update({img: "post" for img in post_imgs})

        # Dot trials
        dot_imgs = pre_imgs + post_imgs
        dot_trials = generate_dot_trials(dot_imgs, phase_lookup, image_info, block=block_idx)
        random.shuffle(dot_trials)
        all_trials.extend(dot_trials)

        # Recognition trials (except final block)
        if block_idx < n_target_sets:
            recog_trials = generate_recognition_trials(pre_imgs, post_imgs, image_info, block=block_idx)
            all_trials.extend(recog_trials)

        previous_target_set = target_pre
        previous_catch_set = catch_pre
        
        block_num += 1
    
    # Final recognition block (only post of last target/catch set)
    final_recog_trials = generate_recognition_trials([], post_imgs, image_info, block=block_num)
    for trial in final_recog_trials:
        trial["block"] = block_num-1
    all_trials.extend(final_recog_trials)

    df = pd.DataFrame(all_trials)
    df["trial_number"] = range(1, len(df) + 1)
    cols = ["trial_number"] + [col for col in df.columns if col != "trial_number"]
    df = df[cols]


    # === Summary info ===
    
    print("=== EXPERIMENT STRUCTURE SUMMARY ===")

    print(f"Total blocks: {blocks_total} (last one is dot-only)")
    print(f"Total trials: {len(df)}")


    if verbose:
        print(f"Total labels loaded: {len(labels)}")
        print(f"Target images: {n_target_sets * images_per_set}")
        print(f"Catch images: {int(n_target_sets * images_per_set * catch_ratio)}")
        print(f"Dot trials per image: {dot_trials_per_image}")
        print("\nManipulation counts (unique images):")
        img_df = pd.DataFrame.from_dict(image_info, orient='index')
        print(img_df.groupby(["image_type", "manipulation"]).size())
        print("\nNUMBER OF TRIALS PER CONDITION:")
        print(df.groupby(["block_type", "image_type", "phase", "manipulation"]).size())

    return df



# ==== FITNESS FUNCTION ====

def fitness_full(sequence):
    violations = 0

    for i in range(len(sequence) - 1):
        curr = sequence[i]
        next_trial = sequence[i + 1]

        # Constraint 1: gray immediately after mooney of same image
        if (
            curr["image_id"] == next_trial["image_id"]
            and curr["phase"] in ["pre", "post"]
            and next_trial["phase"] == "gray"
        ):
            violations += 10

        # Constraint 2: repeated image_id in sequence
        if curr["image_id"] == next_trial["image_id"]:
            violations += 5

    # Constraint 3: triple same manipulation
    for i in range(len(sequence) - 2):
        a, b, c = sequence[i], sequence[i + 1], sequence[i + 2]
        if a["manipulation"] == b["manipulation"] == c["manipulation"]:
            violations += 1

    return violations

# ==== GENETIC ALGORITHM CORE ====

def evolve_sequence(initial_sequence, fitness_fn, generations=100, population_size=30, elite_size=5, mutation_rate=0.2):
    population = [random.sample(initial_sequence, len(initial_sequence)) for _ in range(population_size)]
    best_sequence = None
    best_fitness = float("inf")

    for gen in range(generations):
        scored = [(seq, fitness_fn(seq)) for seq in population]
        scored.sort(key=lambda x: x[1])
        population = [seq for seq, _ in scored]

        if scored[0][1] < best_fitness:
            best_fitness = scored[0][1]
            best_sequence = scored[0][0]

        if best_fitness == 0:
            break  # found optimal

        # Next generation
        next_pop = population[:elite_size]
        while len(next_pop) < population_size:
            parent = random.choice(population[:10])
            child = mutate_sequence(parent, mutation_rate)
            next_pop.append(child)

        population = next_pop

    return best_sequence

def mutate_sequence(sequence, mutation_rate):
    seq = sequence.copy()
    for _ in range(int(len(seq) * mutation_rate)):
        i, j = random.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]
    return seq

# ==== GRAY APPENDING FIX ====

def append_gray_without_violation(mooney_seq, gray_trials, max_attempts=100):
    for _ in range(max_attempts):
        random.shuffle(gray_trials)
        if not mooney_seq or not gray_trials:
            return mooney_seq + gray_trials
        if mooney_seq[-1]["image_id"] != gray_trials[0]["image_id"]:
            return mooney_seq + gray_trials
    raise ValueError("Couldn't resolve gray-after-mooney constraint in 100 attempts.")

# ==== BLOCK-LEVEL OPTIMIZER ====

def optimize_session(df, fitness_fn, generations=100, population_size=30, elite_size=5, mutation_rate=0.2, verbose=True):
    optimized_blocks = []

    for block_num in sorted(df["block"].unique()):
        block_df = df[df["block"] == block_num].copy()

        dot_trials = block_df[block_df["block_type"] == "dot"].to_dict("records")
        rec_df = block_df[block_df["block_type"] == "recognition"]
        mooney_trials = rec_df[rec_df["phase"].isin(["pre", "post"])].to_dict("records")
        gray_trials = rec_df[rec_df["phase"] == "gray"].to_dict("records")

        if VERBOSE_LEVEL in ["full"]:
            print(f"\n🔁 Optimizing block {block_num}...")
            print(f"  🔸 {len(dot_trials)} dot trials")
            print(f"  🔹 {len(mooney_trials)} mooney recognition trials")
            print(f"  🔸 {len(gray_trials)} gray trials")

        optimized_block = []

        if dot_trials:
            optimized_dot = evolve_sequence(dot_trials, fitness_fn, generations, population_size, elite_size, mutation_rate)
            optimized_block.extend(optimized_dot)

        if mooney_trials:
            optimized_mooney = evolve_sequence(mooney_trials, fitness_fn, generations, population_size, elite_size, mutation_rate)
            rec_sequence = append_gray_without_violation(optimized_mooney, gray_trials)
            optimized_block.extend(rec_sequence)

        optimized_blocks.extend(optimized_block)

    optimized_df = pd.DataFrame(optimized_blocks)
    optimized_df["trial_number"] = range(1, len(optimized_df) + 1)
    return optimized_df

# ==== VALIDATION REPORT ====

def validate_constraints(df, verbose=True):
    violations = {
        "gray_after_mooney": [],
        "consecutive_image_repeats": [],
        "triple_same_manipulation": []
    }

    trials = df.to_dict("records")

    for i in range(len(trials) - 1):
        curr, next_trial = trials[i], trials[i + 1]

        if (
            curr["block_type"] == "recognition"
            and next_trial["block_type"] == "recognition"
            and curr["image_id"] == next_trial["image_id"]
            and curr["phase"] in ["pre", "post"]
            and next_trial["phase"] == "gray"
        ):
            violations["gray_after_mooney"].append((curr["trial_number"], next_trial["trial_number"], curr["image_id"]))

        if curr["image_id"] == next_trial["image_id"]:
            violations["consecutive_image_repeats"].append((curr["trial_number"], next_trial["trial_number"], curr["image_id"]))

    for i in range(len(trials) - 2):
        a, b, c = trials[i], trials[i + 1], trials[i + 2]
        if a["manipulation"] == b["manipulation"] == c["manipulation"]:
            violations["triple_same_manipulation"].append(
                (a["trial_number"], b["trial_number"], c["trial_number"], a["manipulation"])
            )

    if VERBOSE_LEVEL in ["full"]:
        print("\n🔍 CONSTRAINT VALIDATION SUMMARY")
        for k, v in violations.items():
            print(f"  ❌ {k}: {len(v)} violations")
            if len(v) > 0:
                print("     Examples:", v[:3])

        if all(len(v) == 0 for v in violations.values()):
            print("✅ No constraint violations detected.")

    return violations


# === MAIN ===
MAX_RETRIES = 10  # just in case

if __name__ == "__main__":
    df = generate_full_session("image_files.txt")
    
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n⚙️ Attempt {attempt}: optimizing trial order...")
        optimized_df = optimize_session(df, fitness_fn=fitness_full)
        violations = validate_constraints(optimized_df, verbose=True)

        if all(len(v) == 0 for v in violations.values()):
            print("✅ Constraints satisfied. Saving final design.")
            optimized_df.to_csv("optimized_experiment_session.csv", index=False)
            break
        else:
            print("❌ Violations found — retrying optimization...")
    else:
        print("⚠️ Max attempts reached. Some violations remain.")
        
    optimized_df.to_csv("optimized_experiment_session.csv", index=False)
# === MULTI-SUBJECT SUPPORT ===

def run_subject(subject_id, MAX_RETRIES=10):
    print(f" \n\n")
    print(f"🧠 Starting: Subject {subject_id}!")
    df = generate_full_session("image_files.txt")
    df["subject"] = subject_id

    for attempt in range(MAX_RETRIES):
        if VERBOSE_LEVEL in ["full"]:
            print(f"\n🧠 Optimizing subject {subject_id} (attempt {attempt + 1})...")

        optimized_df = optimize_session(df, fitness_fn=fitness_full, verbose=(VERBOSE_LEVEL == "full"))
        violations = validate_constraints(optimized_df, verbose=(VERBOSE_LEVEL != "none"))

        if all(len(v) == 0 for v in violations.values()):
            os.makedirs("../designs", exist_ok=True)
            filename = f"../designs/subject_{subject_id}.csv"
            optimized_df.to_csv(filename, index=False)
            print(f"✅ Subject {subject_id} saved to {filename}")
            return

    os.makedirs("../designs", exist_ok=True)
    filename = f"../designs/subject_{subject_id}.csv"
    optimized_df.to_csv(filename, index=False)
    print(f"⚠️ Subject {subject_id} failed optimization after {MAX_RETRIES} attempts.")

    print(f"✅ Subject {subject_id} saved to {filename} nevertheless...")

if __name__ == "__main__":
    for subj in range(1, 6):  # Generate for subjects 1 to 5
        run_subject(subj, optimization_attemps)
