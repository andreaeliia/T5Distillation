"""
Script di esempio per eseguire la distillazione T5
Questo script mostra come utilizzare il sistema step by step
"""

import subprocess
import sys
import os
import json

def run_command(cmd, description=""):
    """Esegue un comando e gestisce errori"""
    print(f"\n{'='*50}")
    print(f"ESEGUENDO: {description}")
    print(f"Comando: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESSO!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ ERRORE!")
        print(f"Codice errore: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def create_custom_config():
    """Crea una configurazione personalizzata per test veloce"""
    config = {
        "model_config": {
            "teacher_model_name": "t5-large",
            "student_model_name": "t5-small",
            "max_input_length": 256,  # Ridotto per test più veloce
            "max_target_length": 64,   # Ridotto per test più veloce
            "cache_dir": "./models_cache"
        },
        "data_config": {
            "dataset_name": "cnn_dailymail",
            "dataset_version": "3.0.0",
            "train_size": 100,        # Molto ridotto per test
            "val_size": 20,           # Molto ridotto per test
            "test_size": 10,          # Molto ridotto per test
            "cache_dir": "./data_cache",
            "num_workers": 2          # Ridotto per evitare problemi
        },
        "distillation_config": {
            "temperature": 4.0,
            "alpha": 0.7,
            "beta": 0.3,
            "kl_loss_weight": 1.0
        },
        "training_config": {
            "batch_size": 2,          # Ridotto per hardware limitato
            "learning_rate": 5e-5,
            "num_epochs": 1,          # Solo 1 epoca per test
            "warmup_steps": 50,       # Ridotto
            "logging_steps": 10,      # Log più frequente
            "save_steps": 100,
            "eval_steps": 50,         # Eval più frequente
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
            "output_dir": "./test_distilled_model",
            "logs_dir": "./test_logs",
            "checkpoint_dir": "./test_checkpoints"
        },
        "system_config": {
            "device": "cuda",         # Cambia in "cpu" se non hai GPU
            "mixed_precision": True,
            "dry_run": False,
            "seed": 42,
            "log_level": "INFO"
        }
    }
    
    config_path = "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configurazione di test creata: {config_path}")
    return config_path

def main():
    print("🚀 Script di esempio per T5 Distillation")
    print("Questo script ti guiderà attraverso l'uso del sistema")
    
    # Controlla se siamo in un ambiente virtuale
    if sys.prefix == sys.base_prefix:
        print("\n⚠️  ATTENZIONE: Non sembra che tu sia in un ambiente virtuale!")
        print("Si consiglia di creare e attivare un venv:")
        print("python -m venv venv")
        print("source venv/bin/activate  # Linux/Mac")
        print("venv\\Scripts\\activate     # Windows")
        response = input("\nVuoi continuare comunque? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n📋 STEP 1: Verifica dipendenze")
    try:
        import torch
        import transformers
        import datasets
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Transformers: {transformers.__version__}")
        print(f"✅ Datasets: {datasets.__version__}")
        print(f"✅ CUDA disponibile: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ Dipendenza mancante: {e}")
        print("Esegui: pip install -r requirements.txt")
        return
    
    print("\n📋 STEP 2: Crea configurazione di default")
    if not run_command([sys.executable, "main.py", "--create-config"], 
                      "Creazione configurazione di default"):
        return
    
    print("\n📋 STEP 3: Test dry-run")
    print("Questo testerà il codice senza training reale...")
    if not run_command([sys.executable, "main.py", "--dry-run", "--log-level", "INFO"], 
                      "Test dry-run della pipeline"):
        print("❌ Il dry-run è fallito. Controlla gli errori sopra.")
        return
    
    print("\n📋 STEP 4: Configurazione per test veloce")
    test_config = create_custom_config()
    
    print("\n📋 STEP 5: Training di test")
    print("Questo eseguirà un training reale ma molto ridotto...")
    print("⚠️  Questo scaricherà i modelli T5 (~3GB) e il dataset CNN/DailyMail")
    
    response = input("\nVuoi procedere con il training di test? (y/n): ")
    if response.lower() == 'y':
        if not run_command([sys.executable, "main.py", "--config", test_config, "--log-level", "INFO"], 
                          "Training di test con dataset ridotto"):
            print("❌ Il training di test è fallito.")
            return
        
        print("\n✅ TRAINING COMPLETATO!")
        print("\nControlla i risultati in:")
        print("- ./test_distilled_model/ (modelli salvati)")
        print("- ./test_logs/ (log e metriche)")
        print("- ./test_checkpoints/ (checkpoint intermedi)")
    
    print("\n📋 STEP 6: Prossimi passi")
    print("""
Per training completo:
1. Modifica distillation_config.json:
   - Aumenta train_size, val_size (o usa null per dataset completo)
   - Aumenta num_epochs (3-5 per risultati buoni)
   - Regola batch_size in base alla tua GPU
   
2. Esegui training completo:
   python main.py
   
3. Monitora i log in tempo reale:
   tail -f logs/training.log
   
4. Per parametri avanzati, vedi README.md

🎉 Setup completato con successo!
    """)

if __name__ == "__main__":
    main()