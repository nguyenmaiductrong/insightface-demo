import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve
import argparse
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_cfg, get_app, detect_faces, face_to_embedding, l2_normalize, load_image_as_bgr


def compute_similarity_matrix_from_facebank(facebank_dir: Path, cfg: dict) -> tuple[np.ndarray, list[str]]:
    print("Computing similarity matrix from facebank...")
    
    app = get_app(cfg)
    
    image_paths = []
    person_labels = []
    embeddings = []
    
    for person_dir in sorted(facebank_dir.iterdir()):
        if not person_dir.is_dir():
            continue
            
        person_name = person_dir.name
        person_images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
        
        print(f"Processing {person_name}: {len(person_images)} images")
        
        for img_path in person_images:
            img_bgr = load_image_as_bgr(img_path)
            if img_bgr is None:
                continue
                
            faces = detect_faces(app, img_bgr)
            if not faces:
                continue
                
            face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
            emb = face_to_embedding(face)
            if emb is None:
                continue
                
            emb = l2_normalize(emb.reshape(1, -1))[0]
            
            embeddings.append(emb)
            image_paths.append(f"{person_name}/{img_path.name}")
            person_labels.append(person_name)
    
    if not embeddings:
        raise ValueError("No valid embeddings found in facebank!")
    
    embeddings = np.vstack(embeddings)
    similarity_matrix = np.dot(embeddings, embeddings.T)  # Cosine similarity (since normalized)
    
    print(f"Computed {similarity_matrix.shape} similarity matrix from {len(set(person_labels))} persons")
    return similarity_matrix, image_paths

def extract_person_from_path(path: str) -> str:
    return path.split('/')[0] if '/' in path else path.split('\\')[0]

def generate_pairs_labels(similarity_matrix: np.ndarray, image_paths: list[str]) -> tuple[list[float], list[int]]:
    similarities = []
    ground_truth = []
    
    persons = [extract_person_from_path(path) for path in image_paths]
    n = len(image_paths)
    
    for i in range(n):
        for j in range(i + 1, n):
            sim_score = similarity_matrix[i, j]
            is_same_person = 1 if persons[i] == persons[j] else 0
            
            similarities.append(sim_score)
            ground_truth.append(is_same_person)
    
    return similarities, ground_truth

def find_optimal_threshold(similarities: list[float], labels: list[int]) -> dict:
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    precision, recall, thresholds_pr = precision_recall_curve(labels, similarities)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_pr[optimal_idx]
    
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    predictions = (similarities >= optimal_threshold).astype(int)
    accuracy = (predictions == labels).mean()
    
    return {
        'threshold': optimal_threshold,
        'f1_score': optimal_f1,
        'precision': optimal_precision,
        'recall': optimal_recall,
        'accuracy': accuracy,
        'precision_curve': precision,
        'recall_curve': recall,
        'thresholds': thresholds_pr
    }

def plot_threshold_analysis(results: dict, output_dir: Path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    f1_scores = 2 * (results['precision_curve'] * results['recall_curve']) / (results['precision_curve'] + results['recall_curve'] + 1e-8)
    ax.plot(results['thresholds'], f1_scores[:-1], 'b-', linewidth=2, label='F1-Score')
    ax.axvline(x=results['threshold'], color='red', linestyle='--', linewidth=2, 
               label=f'Optimal Threshold = {results["threshold"]:.3f}')
    ax.axhline(y=results['f1_score'], color='red', linestyle=':', alpha=0.7,
               label=f'Max F1-Score = {results["f1_score"]:.3f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score vs Threshold - Optimal Threshold Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Calibrate face recognition threshold')
    parser.add_argument('--facebank-dir', type=str, default='data/facebank',
                       help='Path to facebank directory')
    parser.add_argument('--config', type=str, default='src/faceid/configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for plots and results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    facebank_dir = Path(args.facebank_dir)
    if not facebank_dir.exists():
        print(f"Facebank directory not found: {facebank_dir}")
        return
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    cfg = load_cfg(config_path)
    
    similarity_matrix, image_paths = compute_similarity_matrix_from_facebank(facebank_dir, cfg)
    
    print(f"Matrix size: {similarity_matrix.shape}")
    print(f"Unique persons: {len(set(extract_person_from_path(p) for p in image_paths))}")
    
    similarities, labels = generate_pairs_labels(similarity_matrix, image_paths)
    print(f"Generated {len(similarities)} pairs ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
    
    results = find_optimal_threshold(similarities, labels)
    
    print(f"\nTHRESHOLD OPTIMIZATION RESULTS")
    print(f"{'='*50}")
    print(f"Current threshold:     0.36")
    print(f"Optimal threshold:     {results['threshold']:.4f}")
    print(f"{'='*50}")
    print(f"PERFORMANCE METRICS:")
    print(f"Accuracy:              {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"Precision:             {results['precision']:.4f} ({results['precision']*100:.1f}%)")
    print(f"Recall:                {results['recall']:.4f} ({results['recall']*100:.1f}%)")
    print(f"F1-Score:              {results['f1_score']:.4f} ({results['f1_score']*100:.1f}%)")
    print(f"{'='*50}")
    
    if results['threshold'] < 0.36:
        print(f"RECOMMENDATION: Lower threshold from 0.36 to {results['threshold']:.4f}")
        print("BENEFIT: Better recall (catch more true matches)")
    elif results['threshold'] > 0.36:
        print(f"RECOMMENDATION: Raise threshold from 0.36 to {results['threshold']:.4f}")
        print("BENEFIT: Better precision (fewer false matches)")
    else:
        print("RECOMMENDATION: Current threshold is already optimal!")
    
    plot_threshold_analysis(results, output_dir)
    
    with open(output_dir / 'optimal_threshold.txt', 'w') as f:
        f.write("OPTIMAL THRESHOLD CONFIGURATION\n")
        f.write("=" * 40 + "\n")
        f.write(f"threshold: {results['threshold']:.4f}\n")
        f.write(f"\nPerformance:\n")
        f.write(f"- Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"- Precision: {results['precision']:.4f}\n")
        f.write(f"- Recall:    {results['recall']:.4f}\n")
        f.write(f"- F1-Score:  {results['f1_score']:.4f}\n")
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Chart saved to {output_dir}/threshold_analysis.png")
    print(f"Config saved to {output_dir}/optimal_threshold.txt")

if __name__ == "__main__":
    main()
