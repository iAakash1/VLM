# scripts/01_prepare_dataset.py
"""
PlantDx — Step 1: Dataset Preparation

Reads the raw PlantVillage directory and produces:
  data/captions/captions.json  — full {image, caption, plant, condition, class_name} records
  data/splits/splits.json      — {"train": [...], "val": [...]} split

Run from the project root (PlantDx/):
    python scripts/01_prepare_dataset.py

No arguments needed — all paths come from configs/config.yaml.
"""

import sys
import json
import random
import re
from pathlib import Path

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import Paths, load_config
from utils.io_utils import save_json, VALID_EXTENSIONS


# ── 10 caption templates per disease ─────────────────────────────────────────

TEMPLATES = {
    "late_blight": [
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. The leaf surface shows large water-soaked lesions with dark brown centers and pale green halos, indicating rapid tissue collapse from late blight infection.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Dark irregular blotches with greasy water-soaked margins are spreading across the leaf, characteristic of Phytophthora late blight.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Significant portions of the leaf show dark necrotic patches with pale borders and possible white sporulation on the underside.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. The leaf exhibits rapidly expanding brown-black lesions with faint green halos, consistent with advanced late blight progression.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Greasy dark lesions with irregular edges and surrounding chlorotic tissue are visible across this {plant} leaf.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Water-soaked spots transitioning to dark necrotic tissue with pale margins dominate the leaf surface, showing late blight damage.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Large coalescing dark patches with pale green to yellow borders indicate severe late blight affecting this {plant} leaf.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. The leaf shows dark brown to black irregular lesions spreading inward from the margins, typical of Phytophthora infestans infection.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Collapsed water-soaked tissue forming dark patches with pale surrounding zones is visible on this infected {plant} leaf.",
        "Plant: {plant}. Condition: Late Blight. Severity: {sev}. Progressive dark lesions with pale halos and wilted tissue edges across the leaf indicate moderate to severe late blight.",
    ],
    "early_blight": [
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. The leaf shows circular brown lesions with distinctive concentric rings forming a target-like pattern surrounded by yellow chlorotic zones.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Target-shaped dark brown spots with concentric rings and yellow halos are scattered across this {plant} leaf surface.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Bull-eye patterned lesions with dark centers and lighter concentric bands surrounded by yellowing tissue indicate early blight.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Multiple circular spots with dark brown concentric rings and chlorotic margins are visible on this {plant} leaf.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Classic early blight symptoms present as target-like spots with angular dark centers and surrounding yellow tissue on the leaf.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. The leaf displays several necrotic spots with concentric ring patterns typical of Alternaria solani early blight infection.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Dark brown lesions with concentric banding and yellow margins characterize this early blight infection on the {plant} leaf.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Irregular brown spots with lighter concentric zones and yellow surrounding tissue are distributed across the leaf surface.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Target-pattern lesions with dark brown rings and chlorotic halos are a hallmark of the early blight visible on this leaf.",
        "Plant: {plant}. Condition: Early Blight. Severity: {sev}. Circular necrotic areas with concentric dark and light banding surrounded by chlorosis indicate early blight on this {plant}.",
    ],
    "bacterial_spot": [
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Small water-soaked lesions have turned dark brown with distinct yellow halos scattered irregularly across the leaf surface.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. The leaf displays numerous irregular brown spots surrounded by yellow margins, consistent with bacterial spot infection.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Multiple dark lesions with a water-soaked greasy appearance and chlorotic halos are distributed across this {plant} leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Angular brown spots bounded by leaf veins with surrounding yellowing indicate bacterial spot disease on this leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Dark necrotic spots with yellow borders and irregular margins are characteristic of bacterial spot on this {plant} leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. The leaf shows small raised dark lesions with water-soaked centers and chlorotic halos caused by Xanthomonas bacterial infection.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Scattered dark brown spots with yellow surrounding tissue and irregular shapes cover portions of this {plant} leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Water-soaked spots transitioning to brown necrotic lesions with yellow halos are visible across the leaf surface.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. Multiple small angular lesions with dark centers and chlorotic margins indicate bacterial spot infection on this leaf.",
        "Plant: {plant}. Condition: Bacterial Spot. Severity: {sev}. The {plant} leaf exhibits irregularly shaped dark spots with greasy margins and surrounding yellowing from bacterial infection.",
    ],
    "healthy": [
        "Plant: {plant}. Condition: Healthy. Severity: None. The leaf displays uniform vibrant green coloration with no visible lesions, spots, discoloration, or any signs of disease or stress.",
        "Plant: {plant}. Condition: Healthy. Severity: None. This {plant} leaf appears completely healthy showing consistent green pigmentation, smooth texture, and no abnormalities whatsoever.",
        "Plant: {plant}. Condition: Healthy. Severity: None. No disease symptoms are visible on this leaf. Normal healthy growth is indicated by even green color and intact leaf structure.",
        "Plant: {plant}. Condition: Healthy. Severity: None. The {plant} leaf shows excellent health with rich green coloration, no spots or lesions, and normal leaf morphology throughout.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Uniform green coloration and absence of any lesions, spots, or discoloration confirm this {plant} leaf is in a healthy condition.",
        "Plant: {plant}. Condition: Healthy. Severity: None. The leaf surface is clean and uniformly green with no visible fungal growth, bacterial lesions, or viral symptoms present.",
        "Plant: {plant}. Condition: Healthy. Severity: None. Healthy leaf tissue is indicated by consistent green pigmentation and smooth surface with no necrotic or chlorotic areas visible.",
        "Plant: {plant}. Condition: Healthy. Severity: None. This {plant} leaf is free from any disease symptoms showing normal venation, green pigmentation, and intact margins.",
        "Plant: {plant}. Condition: Healthy. Severity: None. No pathological symptoms observed. The leaf exhibits vigorous healthy appearance with uniform coloration and no damage.",
        "Plant: {plant}. Condition: Healthy. Severity: None. The {plant} leaf is in optimal health showing deep green color, no discoloration, and complete absence of disease indicators.",
    ],
    "leaf_mold": [
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pale yellow spots on the upper leaf surface correspond to olive-green velvety fungal growth on the underside.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. The leaf shows chlorotic patches above with a distinctive olive-colored mold coating beneath, indicating Passalora leaf mold.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Yellow to pale green discoloration on the upper surface with corresponding fungal sporulation below characterizes this leaf mold infection.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Irregular yellow blotches on the upper leaf surface and velvety olive-green growth on the lower surface indicate leaf mold.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. The {plant} leaf exhibits chlorotic spots above with a felt-like olive-brown fungal coating on the corresponding lower surface.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Pale yellow patches visible from above and dense fungal sporulation beneath indicate moderate leaf mold infection on this leaf.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Upper leaf surface shows faint yellowing while the underside displays characteristic olive-green mold growth from Passalora fulva.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Chlorotic areas on the adaxial surface with corresponding abaxial fungal growth are visible on this {plant} leaf.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. The leaf displays yellow irregular spots on top and velvety dark sporulation beneath, consistent with leaf mold disease.",
        "Plant: {plant}. Condition: Leaf Mold. Severity: {sev}. Diffuse yellowing on the upper surface and olive-green mold patches on the underside characterize this leaf mold infection.",
    ],
    "septoria": [
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Small circular spots with dark brown borders, lighter tan centers, and tiny black pycnidia dots are scattered across the leaf.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Numerous small lesions with dark margins and lighter centers containing visible black fruiting bodies characterize this infection.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Multiple circular spots with pycnidia at their centers are spread across the leaf, consistent with Septoria lycopersici.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. The leaf shows many small round spots with dark borders, gray-tan centers, and minute black specks within each lesion.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Tiny circular lesions with dark edges and pale tan centers containing black pycnidia are distributed across this {plant} leaf.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Characteristic septoria spots appear as small circles with dark brown margins and gray centers dotted with pycnidia.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. The {plant} leaf displays numerous small round lesions with dark borders and lighter centers showing black fruiting bodies.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Small circular necrotic spots with dark halos and tan interiors containing pycnidia indicate septoria leaf spot.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Dense spots with dark rings and pale centers scattered across the leaf surface indicate septoria infection.",
        "Plant: {plant}. Condition: Septoria Leaf Spot. Severity: {sev}. Round lesions with dark brown edges and grayish centers with visible black pycnidia are spread across this leaf.",
    ],
    "spider_mites": [
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Fine stippling and yellow speckling cover the leaf surface with delicate webbing possibly visible between structures.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. The leaf shows a bronzed dusty appearance with tiny yellow feeding dots from two-spotted spider mite damage.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Extensive stippling creates a silvery mottled pattern across the {plant} leaf from spider mite feeding activity.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Thousands of tiny puncture marks cause a speckled yellow-bronze discoloration across the upper leaf surface.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. The upper leaf surface appears pale and stippled with yellow dots while fine silk webbing may be present beneath.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Mite feeding has caused a characteristic salt-and-pepper stippled pattern with overall leaf bronzing visible.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. The {plant} leaf shows chlorotic stippling and bronzing from Tetranychus urticae feeding across the leaf surface.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Diffuse yellow speckling and a dusty bronzed appearance across the leaf indicate significant spider mite infestation.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. Fine yellow dots and possible webbing from two-spotted spider mites are distributed across this {plant} leaf.",
        "Plant: {plant}. Condition: Spider Mite Damage. Severity: {sev}. The leaf surface displays pale stippling and slight bronzing consistent with moderate to heavy spider mite damage.",
    ],
    "target_spot": [
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Brown circular lesions with concentric rings and yellow surrounding tissue form a distinct target pattern on the leaf.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Ringed brown spots with chlorotic margins indicate target spot disease caused by Corynespora cassiicola.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Distinct bull-eye patterned lesions surrounded by yellow tissue are visible across the infected {plant} leaf.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Circular brown spots with concentric lighter zones and chlorotic halos are scattered on this leaf surface.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. The leaf displays target-like spots with alternating dark and light concentric bands surrounded by yellowing tissue.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Multiple ringed lesions with dark centers and lighter concentric zones surrounded by chlorosis indicate target spot.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. The {plant} leaf shows circular necrotic spots with ring patterns and yellow margins from target spot infection.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Dark centered bull-eye spots with concentric lighter rings and chlorotic borders are present on this leaf.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Characteristic ringed lesions resembling a target with brown bands and yellow halos cover portions of this leaf.",
        "Plant: {plant}. Condition: Target Spot. Severity: {sev}. Circular lesions with alternating dark and light concentric bands and yellow surrounding areas indicate target spot.",
    ],
    "mosaic_virus": [
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. An irregular mosaic pattern of alternating light and dark green patches with leaf distortion and crumpling is visible.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. The leaf shows a mottled appearance with alternating yellow-green and dark green zones and slight distortion.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Systemic viral infection has caused mosaic discoloration with irregular green islands and leaf surface distortion.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Characteristic mosaic symptoms appear as irregular light and dark green mottling with mild leaf curling on this {plant}.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. The {plant} leaf displays a patchwork of pale and dark green areas with slight wrinkling from mosaic virus.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Irregular alternating light and dark green mosaic pattern with leaf distortion indicates viral mosaic infection.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. The leaf surface shows chlorotic and dark green mosaic patterning with slight rugosity from viral infection.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Mosaic discoloration with intermingled pale and deep green patches and mild leaf deformation is present on this leaf.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. Systemic mosaic virus symptoms including mottled coloration and leaf distortion are evident on this {plant} leaf.",
        "Plant: {plant}. Condition: Mosaic Virus. Severity: {sev}. The leaf displays irregular green mosaic patterning with alternating chlorotic patches from viral mosaic infection.",
    ],
    "yellow_leaf_curl": [
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Severe upward curling of leaf margins with intense yellowing and stunted crinkled appearance throughout the leaf.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. The leaf is severely distorted with upward cupping edges and prominent chlorosis typical of TYLCV infection.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Extreme chlorosis and upward leaf cupping with crinkled texture indicate yellow leaf curl viral infection.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. The {plant} leaf shows severe yellowing and upward curling with reduced size from yellow leaf curl virus.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Strong upward leaf rolling with intense yellowing and stunted growth characterize this TYLCV infection.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. The leaf margins curl upward severely with chlorotic discoloration and distorted crinkled surface from viral infection.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Yellowing and upward cupping of the {plant} leaf with small leaf size indicate yellow leaf curl virus.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Severe chlorosis combined with upward leaf curling and reduced blade size are visible on this infected leaf.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. The leaf displays intense yellowing and strong upward rolling with distorted surface texture from TYLCV.",
        "Plant: {plant}. Condition: Yellow Leaf Curl Virus. Severity: {sev}. Upward leaf curl and bright yellow discoloration with crinkled texture indicate severe yellow leaf curl virus.",
    ],
}

SEVERITY_MAP = {
    "bacterial_spot":   "Moderate to Severe",
    "early_blight":     "Mild to Moderate",
    "late_blight":      "Moderate to Severe",
    "leaf_mold":        "Mild",
    "septoria":         "Moderate",
    "spider_mites":     "Mild to Moderate",
    "target_spot":      "Moderate",
    "mosaic_virus":     "Severe",
    "yellow_leaf_curl": "Severe",
    "yellowleaf":       "Severe",
    "healthy":          "None",
}

FALLBACK_TEMPLATES = [
    "Plant: {plant}. Condition: {cond}. Severity: {sev}. The leaf shows visible disease symptoms including discoloration and tissue damage consistent with {cond} infection.",
    "Plant: {plant}. Condition: {cond}. Severity: {sev}. This leaf exhibits signs of {cond} with abnormal coloration and visible lesions present on the surface.",
    "Plant: {plant}. Condition: {cond}. Severity: {sev}. Disease symptoms including lesions and discoloration are present on this {plant} leaf, consistent with {cond}.",
    "Plant: {plant}. Condition: {cond}. Severity: {sev}. The {plant} leaf displays characteristic symptoms of {cond} with visible tissue damage and abnormal pigmentation.",
    "Plant: {plant}. Condition: {cond}. Severity: {sev}. Visible signs of {cond} including discoloration and lesions are observed on the surface of this leaf.",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_folder_name(folder_name: str):
    """
    PlantVillage folders use mixed separators:
      Tomato_Leaf_Mold          (single underscore)
      Potato___Early_blight     (triple underscore)
      Pepper__bell___Bacterial_spot (double + triple)

    Strategy: split on the longest separator first.
    Returns (plant_str, condition_str).
    """
    if "___" in folder_name:
        parts = folder_name.split("___", 1)
    elif "__" in folder_name:
        parts = folder_name.split("__", 1)
    else:
        parts = folder_name.split("_", 1)

    plant     = re.sub(r"_+", " ", parts[0]).strip()
    condition = re.sub(r"_+", " ", parts[1]).strip() if len(parts) > 1 else "Unknown"
    return plant, condition


def get_severity(key: str) -> str:
    for sev_key, sev_val in SEVERITY_MAP.items():
        if sev_key in key:
            return sev_val
    return "Moderate"


def get_templates(key: str) -> list:
    for map_key, tmpl_list in TEMPLATES.items():
        if map_key in key or key in map_key:
            return tmpl_list
    return FALLBACK_TEMPLATES


def build_caption(plant: str, condition: str, img_index: int) -> str:
    key       = condition.lower().replace(" ", "_")
    sev       = get_severity(key)
    templates = get_templates(key)
    template  = templates[img_index % len(templates)]
    return template.format(plant=plant, cond=condition, sev=sev)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg   = load_config()
    paths = Paths(cfg)
    paths.makedirs()

    dataset_path = paths.dataset_root
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Update 'paths.dataset_root' in configs/config.yaml and re-run.")
        sys.exit(1)

    train_ratio = cfg["training"]["train_split"]
    seed        = cfg["training"]["seed"]

    class_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])
    print(f"Found {len(class_folders)} class folders in {dataset_path}\n")

    records = []
    for folder in class_folders:
        plant, condition = parse_folder_name(folder.name)
        images = [f for f in folder.iterdir() if f.suffix in VALID_EXTENSIONS]
        if not images:
            print(f"  WARNING: No images in {folder.name} — skipped")
            continue

        for idx, img_path in enumerate(sorted(images)):
            records.append({
                "image":      str(img_path.resolve()),
                "caption":    build_caption(plant, condition, idx),
                "plant":      plant,
                "condition":  condition,
                "class_name": folder.name,
            })

        n_templates = min(len(images), len(get_templates(condition.lower().replace(" ", "_"))))
        print(f"  {folder.name:<50}  {len(images):>5} images  {n_templates} templates")

    random.seed(seed)
    random.shuffle(records)
    split = int(len(records) * train_ratio)
    train = records[:split]
    val   = records[split:]

    # Save captions (full list, unsplit)
    save_json(records, paths.captions_file)

    # Save splits separately
    save_json({"train": train, "val": val}, paths.splits_file)

    print(f"\n{'='*55}")
    print(f"Total images : {len(records)}")
    print(f"Train        : {len(train)}")
    print(f"Val          : {len(val)}")
    print(f"Captions  →  {paths.captions_file}")
    print(f"Splits    →  {paths.splits_file}")

    # Variation sanity check
    print(f"\nCaption variation check — Late Blight (first 5):")
    lb = [r for r in records if "late_blight" in r["class_name"].lower()][:5]
    unique = len({r["caption"] for r in lb})
    for i, r in enumerate(lb):
        print(f"  [{i+1}] {r['caption'][:85]}...")
    if unique < len(lb):
        print(f"  WARNING: Only {unique}/{len(lb)} unique — check templates")
    else:
        print(f"  OK: {unique}/{len(lb)} unique captions")


if __name__ == "__main__":
    main()
