# Hausarbeit: Vergleich von PyTorch-Trainingsframeworks

## Mögliche Frameworks

1. Ignite (https://github.com/pytorch/ignite)
2. Poutyne (https://poutyne.org)
3. PyTorch Lightning (https://github.com/PyTorchLightning/pytorch-lightning)
4. Determined (https://github.com/determined-ai/determined)
5. Skorch (https://github.com/skorch-dev/skorch)
6. PyTorch-NLP (https://pytorchnlp.readthedocs.io/en/latest/)
7. Fairscale (https://github.com/facebookresearch/fairscale)
8. FaistAI (https://github.com/fastai/fastai)
9. Accelerate
10. DeepSpeed (insbesondere die Transformer-Erweiterung)
11. (sacred)

## Frameworks für Transformers

1. Huggingface Trainer (https://huggingface.co/transformers/main_classes/trainer.html)
2. Simple Transformers (https://github.com/ThilinaRajapakse/simpletransformers)

## Methodik

Beschreibung der Frameworks.

* Designphilosophie
* Zielgruppe/ Gebiet
* API-Struktur beschreiben
* Ggf. Beschreiben von konzeptionellen Vorbildern etc

### Praktisches Experiment

* Training eines neuronalen Netzwerks: Sprachmodell - Transformers
* Training eines neuronalen Netzwerks: From Scratch Python

#### Kriterien

__Funktionen:__

* Vor- und Nachteile der Design-Philosophie und des generellen Aufbaus
* Anpassbarkeit
* Supervised Unsupervised
* Überwachung des Trainings
* Checkpoints während des Trainings
* Fähigkeit zu Distributed Training
* (Fähigkeit mit halber Genauigkeit zu rechnen (FP16))
* (Kommandozeilen Interface)
* (Verwaltung und Vergleich mehrer Durchläufe)
* Metadatenhandling

__Weitere Kriterien:__

* Github Statistiken (Issues, Forks, Commits, etc.)
* Ggf. andere Verbreitungsmetriken:
  * Suche danach in Arxiv (?)
  * Volltextsuche: http://search.arxiv.org:8081/?query=Tensorflow&in=grp_cs
  * Suche nach Projekten in Github, die Framework X einsetzen

# Neuer Plan:

* Fokus auf wenige Frameworks, die alle im NLP (Transformer) Bereich liegen
* Praktischer Vergleich dieser Frameworks anhand von 2 praktischen Projekten.
  * Training eines Standard-Modells
  * Training eines eigenen Modells
* Hauptfokus auf PyTorch aber auch Hinweise auf die Arbeit mit Tensorflow (insbesondere Keras)
* 
