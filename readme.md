# T5 Model Distillation

Questo progetto implementa la distillazione di conoscenza per trasferire le capacità di T5-Large (770M parametri) a T5-Small (60M parametri) per compiti di text generation, specificamente summarization utilizzando il dataset CNN/DailyMail.

## Struttura del Progetto

```
├── config.py           # Gestione configurazione e parametri
├── dataloader.py       # Caricamento e preprocessing dei dati  
├── model.py            # Gestione modelli teacher e student
├── distillation.py     # Logica di distillazione e loss functions
├── trainer.py          # Loop di training principale
├── main.py             # Script principale
├── requirements.txt    # Dipendenze Python
└── README.md          # Questo file
```

## Installazione

1. **Crea un ambiente virtuale:**
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

2. **Installa le dipendenze:**
```bash
pip install -r requirements.txt
```

## Configurazione

### Creazione Config File
Prima di tutto, genera il file di configurazione di default:

```bash
python main.py --create-config
```

Questo creerà `distillation_config.json` con tutti i parametri configurabili:

```json
{
  "model_config": {
    "teacher_model_name": "t5-large",
    "student_model_name": "t5-small", 
    "max_input_length": 512,
    "max_target_length": 128,
    "cache_dir": "./models_cache"
  },
  "data_config": {
    "dataset_name": "cnn_dailymail",
    "dataset_version": "3.0.0",
    "train_size": 1000,
    "val_size": 200,
    "test_size": 100,
    "cache_dir": "./data_cache",
    "num_workers": 4
  },
  "distillation_config": {
    "temperature": 4.0,
    "alpha": 0.7,
    "beta": 0.3,
    "kl_loss_weight": 1.0
  },
  "training_config": {
    "batch_size": 4,
    "learning_rate": 5e-05,
    "num_epochs": 3,
    "warmup_steps": 500,
    "logging_steps": 100,
    "save_steps": 1000,
    "eval_steps": 500,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "output_dir": "./distilled_model",
    "logs_dir": "./logs",
    "checkpoint_dir": "./checkpoints"
  },
  "system_config": {
    "device": "cuda",
    "mixed_precision": true,
    "dry_run": false,
    "seed": 42,
    "log_level": "INFO"
  }
}
```

### Personalizzazione Config

Puoi modificare `distillation_config.json` per:

- **Dimensioni dataset**: Cambia `train_size`, `val_size`, `test_size` (usa `null` per dataset completo)
- **Parametri di training**: Modifica `batch_size`, `learning_rate`, `num_epochs`
- **Distillation**: Regola `temperature`, `alpha`, `beta` per bilanciare le loss
- **Hardware**: Imposta `device` su "cuda", "cpu", o "mps"

## Utilizzo

### Modalità Test (Dry Run)
Per testare il codice senza training completo:

```bash
python main.py --dry-run
```

Questa modalità:
- Usa modelli minimali per ridurre memoria
- Processa solo pochi campioni del dataset
- Simula il training senza calcoli pesanti
- Perfetta per verificare che tutto funzioni

### Training Completo
Per il training reale:

```bash
python main.py
```

### Opzioni Aggiuntive

```bash
# Specificare config personalizzato
python main.py --config my_config.json

# Cambiare livello di logging
python main.py --log-level DEBUG

# Combinazioni
python main.py --dry-run --log-level DEBUG
```

## Dataset

Il progetto usa **CNN/DailyMail** per summarization:
- **Task**: Generare riassunti di articoli di news
- **Input**: "summarize: [articolo]"  
- **Output**: Highlights dell'articolo
- **Dimensioni**: ~300k esempi di training

Il dataset viene scaricato automaticamente dalla libreria `datasets` di Hugging Face.

## Architettura di Distillazione

### Modelli
- **Teacher**: T5-Large (770M parametri)
- **Student**: T5-Small (60M parametri)
- **Compression ratio**: ~13x

### Loss Function
La loss combina tre componenti:

```
Total Loss = α × Distillation Loss + β × Ground Truth Loss

Distillation Loss = KL_divergence(Student_logits/T, Teacher_logits/T) × T²
Ground Truth Loss = CrossEntropy(Student_logits, True_labels)
```

Dove:
- `T` = temperatura (default: 4.0)
- `α` = peso distillation loss (default: 0.7)  
- `β` = peso ground truth loss (default: 0.3)

## Monitoraggio

Il sistema genera log completi:

- **Console logs**: Progress e metriche in tempo reale
- **File logs**: `./logs/training.log` e `./logs/main.log`
- **Metriche JSON**: `./logs/training_metrics.json` con tutte le metriche

### Metriche Tracciate
- Total loss, distillation loss, ground truth loss
- Learning rate, perplexity
- Training e validation metrics

## Output

Il training produce:

```
./distilled_model/
├── best_model/          # Miglior modello su validation
└── final_model/         # Modello finale

./checkpoints/           # Checkpoint intermedi
└── checkpoint-step-X.pt

./logs/                  # Log e metriche
├── training.log
├── main.log
└── training_metrics.json
```

## Requisiti Hardware

### Minimum (con dry-run)
- RAM: 8GB
- GPU: Non necessaria
- Storage: 2GB

### Consigliato (training completo)
- RAM: 16GB+
- GPU: 8GB+ VRAM (es. RTX 3070, V100)
- Storage: 10GB+ per dataset e modelli

### Per dataset completo
- RAM: 32GB+
- GPU: 16GB+ VRAM (es. A100, RTX 4090)

## Troubleshooting

### Errori Comuni

**CUDA out of memory**:
```bash
# Riduci batch_size nel config
"batch_size": 2
"gradient_accumulation_steps": 8
```

**Dataset troppo grande**:
```bash
# Limita dimensioni nel config
"train_size": 1000,
"val_size": 200
```

**Errori di import**:
```bash
# Reinstalla dipendenze
pip install --upgrade -r requirements.txt
```

### Debug Mode
Per debug dettagliato:

```bash
python main.py --log-level DEBUG --dry-run
```

## Sviluppo

### Test del Codice
```bash
# Test rapido della pipeline
python main.py --dry-run --log-level DEBUG

# Test con dataset minimo
# Modifica config: train_size: 100, val_size: 20
python main.py
```

### Estensioni Possibili
- Aggiungere altre metriche (BLEU, ROUGE)
- Implementare early stopping
- Supporto per altri modelli T5
- Distillazione multi-task
- Integrazione con Weights & Biases

## Note Importanti

1. **Prima esecuzione**: Il download dei modelli e dataset richiede tempo
2. **Memoria**: T5-Large richiede ~3GB di VRAM minimo
3. **Reproducibilità**: Il seed è fissato per risultati consistenti
4. **Dry-run**: Sempre consigliato testare prima con `--dry-run`

## Licenza

Questo progetto è per scopi di ricerca ed educativi. Rispetta le licenze dei modelli T5 e del dataset CNN/DailyMail.