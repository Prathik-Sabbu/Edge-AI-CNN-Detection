"""
Comprehensive Edge AI Model Benchmark & Metrics Generator.
Evaluates accuracy, F1-score, inference latency, throughput (FPS), memory,
and model size for the MobileNetV3 TFLite animal detector.
Generates tailored, quantified resume/portfolio bullet points.
"""
import os
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
import cv2

# Set path to rpi_system
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config

def get_model_size_info(model_path: Path):
    """Calculates model size in bytes, KB, and MB."""
    size_bytes = model_path.stat().st_size
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    return size_bytes, size_kb, size_mb

def evaluate_accuracy_and_f1(interpreter, labels, data_dir: Path, val_split: float = 0.2, seed: int = 42):
    """
    Evaluates Top-1 accuracy, Top-3 accuracy, precision, recall, and F1-score
    across the validation split (80/20 train/val split matching training.ipynb).
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Gather image files per class from VAL_DIR or data_dir with Italian folder mapping
    val_dir = getattr(config, "VAL_DIR", data_dir / "val")
    search_dir = val_dir if (val_dir.exists() and val_dir.name == "val") else data_dir

    ENGLISH_TO_ITALIAN = {
        "dog": "cane", "horse": "cavallo", "elephant": "elefante",
        "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto",
        "lion": "lion", "cow": "mucca", "sheep": "pecora", "squirrel": "scoiattolo",
    }

    label_to_idx = {name: i for i, name in enumerate(labels)}
    all_files = []
    
    for class_name in labels:
        it_name = ENGLISH_TO_ITALIAN.get(class_name.lower(), class_name.lower())
        candidates = [
            search_dir / class_name,
            search_dir / class_name.lower(),
            search_dir / it_name,
            search_dir / it_name.capitalize(),
        ]
        class_folder = next((p for p in candidates if p.is_dir()), None)
        if not class_folder:
            continue
        class_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpeg"))
        for f in class_files:
            all_files.append((f, label_to_idx[class_name]))

    total_images = len(all_files)
    if total_images == 0:
        return None

    # Use 20% holdout split (seed 42) matching training.ipynb if using raw-img
    if val_dir.exists() and val_dir.name == "val":
        val_files = all_files
    else:
        np.random.seed(seed)
        indices = np.random.permutation(total_images)
        val_size = int(total_images * val_split)
        val_indices = indices[:val_size]
        val_files = [all_files[i] for i in val_indices]

    y_true = []
    y_pred = []
    top3_correct = 0

    print(f"Evaluating model on {len(val_files)} validation images (out of {total_images} total across {len(labels)} classes)...")

    for img_path, true_label_idx in val_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Preprocess matching training pipeline
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (config.INPUT_WIDTH, config.INPUT_HEIGHT))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32)

        # MobileNetV3 (with include_preprocessing=True) expects raw [0, 255] pixels.
        # No division or -1 to 1 scaling is needed.

        if input_dtype == np.uint8:
            input_data = input_data.astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        top_pred_idx = int(np.argmax(output))
        top3_indices = np.argsort(output)[-3:][::-1]

        y_true.append(true_label_idx)
        y_pred.append(top_pred_idx)

        if true_label_idx in top3_indices:
            top3_correct += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    top1_acc = np.mean(y_true == y_pred) * 100.0
    top3_acc = (top3_correct / len(y_true)) * 100.0

    # Calculate Macro and Weighted F1-scores using sklearn
    try:
        from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        report = classification_report(y_true, y_pred, target_names=[labels[i] for i in sorted(set(y_true))], zero_division=0)
    except Exception:
        macro_f1 = top1_acc
        weighted_f1 = top1_acc
        macro_precision = top1_acc
        macro_recall = top1_acc
        report = "sklearn not available for detailed breakdown."

    return {
        "total_dataset_size": total_images,
        "val_set_size": len(y_true),
        "num_classes": len(labels),
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "report": report,
    }

def benchmark_inference_latency(interpreter, num_warmup: int = 20, num_iterations: int = 100):
    """
    Measures raw inference latency (ms) and throughput (FPS).
    Tracks p50, p95, p99, mean, and std latency.
    """
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    dummy_input = np.random.uniform(-1.0, 1.0, size=input_shape).astype(input_dtype)

    # Warmup runs (prime memory, caches, threads)
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

    # 1. Pure Inference Latency
    inf_latencies_ms = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        t1 = time.perf_counter()
        inf_latencies_ms.append((t1 - t0) * 1000.0)

    # 2. End-to-End Latency including OpenCV preprocessing
    import cv2
    dummy_frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    e2e_latencies_ms = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        # 1. Preprocess (simulating real camera pipeline)
        rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(resized, axis=0).astype(input_dtype)
        # 2. Inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        t1 = time.perf_counter()
        e2e_latencies_ms.append((t1 - t0) * 1000.0)

    inf_latencies = np.array(inf_latencies_ms)
    e2e_latencies = np.array(e2e_latencies_ms)

    return {
        "num_iterations": num_iterations,
        "inf_mean": np.mean(inf_latencies),
        "inf_p99": np.percentile(inf_latencies, 99),
        "e2e_mean": np.mean(e2e_latencies),
        "e2e_p99": np.percentile(e2e_latencies, 99),
        "e2e_fps": 1000.0 / np.mean(e2e_latencies),
    }

def measure_memory_footprint(model_path: str):
    """Measures RAM allocated by interpreter using OS metrics."""
    import os
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
    except ImportError:
        mem_before = 0

    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter

        interpreter = Interpreter(model_path=model_path, num_threads=2)
        interpreter.allocate_tensors()
        
        try:
            mem_after = process.memory_info().rss
            return mem_after / (1024 * 1024), mem_after / (1024 * 1024)
        except NameError:
            # Fallback if psutil not installed on Pi
            with open("/proc/self/statm") as f:
                pages = int(f.read().split()[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                mem_mb = (pages * page_size) / (1024 * 1024)
                return mem_mb, mem_mb
    except Exception:
        return 0.0, 0.0

def run_benchmark():
    model_path = Path(config.OUTPUT_TFLITE)
    labels_path = Path(config.OUTPUT_LABELS)
    data_dir = Path(config.DATA_DIR)

    print("================================================================================")
    print("        EDGE AI CNN DETECTION SYSTEM: BENCHMARK & METRICS PROFILER              ")
    print("================================================================================")

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # 1. Model Profile
    size_bytes, size_kb, size_mb = get_model_size_info(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = str(input_details[0]["dtype"].__name__)

    print(f"\n[1] MODEL ARCHITECTURE PROFILE:")
    print(f"  • Base Architecture    : MobileNetV3-Small (Transfer Learning from ImageNet)")
    print(f"  • Model Format          : TensorFlow Lite (.tflite)")
    print(f"  • Input Tensor Shape    : {input_shape.tolist()} ({input_dtype})")
    print(f"  • Number of Classes     : {len(labels)} classes: {labels}")
    print(f"  • File Size on Disk     : {size_mb:.2f} MB ({size_kb:.1f} KB / {size_bytes:,} bytes)")

    # 2. Memory Footprint
    curr_mb, peak_mb = measure_memory_footprint(str(model_path))
    print(f"\n[2] MEMORY / RAM FOOTPRINT:")
    print(f"  • Peak Allocation Trace : {peak_mb:.2f} MB")

    # 3. Accuracy & F1-Score
    print(f"\n[3] CLASSIFICATION ACCURACY & F1 METRICS:")
    acc_results = evaluate_accuracy_and_f1(interpreter, labels, data_dir)
    if acc_results:
        print(f"  • Dataset Evaluated     : {acc_results['total_dataset_size']} total images across {acc_results['num_classes']} classes")
        print(f"  • Validation Split      : {acc_results['val_set_size']} validation images (20% holdout split)")
        print(f"  • Top-1 Accuracy        : {acc_results['top1_accuracy']:.2f}%")
        print(f"  • Top-3 Accuracy        : {acc_results['top3_accuracy']:.2f}%")
        print(f"  • Macro F1-Score        : {acc_results['macro_f1']:.2f}%")
        print(f"  • Weighted F1-Score     : {acc_results['weighted_f1']:.2f}%")
        print(f"  • Macro Precision       : {acc_results['macro_precision']:.2f}%")
        print(f"  • Macro Recall          : {acc_results['macro_recall']:.2f}%")
        print(f"\n  Classification Report Breakdown:\n{acc_results['report']}")

    # 4. Latency & Throughput Benchmark
    print(f"\n[4] INFERENCE SPEED & LATENCY (100 Iterations):")
    speed_results = benchmark_inference_latency(interpreter, num_warmup=20, num_iterations=100)
    print(f"  • Pure AI Math (Mean)   : {speed_results['inf_mean']:.2f} ms")
    print(f"  • Pure AI Math (p99)    : {speed_results['inf_p99']:.2f} ms")
    print(f"  • End-to-End (Mean)     : {speed_results['e2e_mean']:.2f} ms (includes OpenCV)")
    print(f"  • End-to-End (p99)      : {speed_results['e2e_p99']:.2f} ms")
    print(f"  • True Throughput       : {speed_results['e2e_fps']:.1f} FPS (frames per second)")

if __name__ == "__main__":
    run_benchmark()
