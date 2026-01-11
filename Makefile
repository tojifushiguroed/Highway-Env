# ============================================================================
# 🛠️ PROJECT AUTOMATION INTERFACE (FINAL)
# ============================================================================
# Goals:
# - Modular & runnable workflow
# - PEP8 support via black/flake8
# - Safe formatting (does NOT touch venv or random dirs)
# - Mac M4 / Apple Silicon friendly

SHELL := /bin/bash

# System python (usually python3 on macOS/Linux)
SYSTEM_PYTHON := python3

# Virtual environment
VENV := venv
BIN := $(VENV)/bin

# Tools inside venv
PY := $(BIN)/python
PIP := $(BIN)/pip
BLACK := $(BIN)/black
FLAKE8 := $(BIN)/flake8
AUTOFLAKE := $(BIN)/autoflake

# Apple Silicon: allow fallback ops on MPS when needed
export PYTORCH_ENABLE_MPS_FALLBACK=1

# ---------------------------------------------------------------------------
# Project source files to format/lint (SAFE LIST: we only touch these)
# ---------------------------------------------------------------------------
SRC := \
	parking_training.py \
	parking_test.py \
	utils.py \
	merge+intersection_train.py \
	test_merge.py \
	test_intersection.py \
	train_racetrack.py \
	racetrack_test.py \
	roundabout_training.py \
	roundabout_test.py \
	highway_train.py \
	highway_test.py

# ---------------------------------------------------------------------------
# PHONY targets
# ---------------------------------------------------------------------------
.PHONY: help setup install clean lint format \
	train_parking test_parking \
	train_racetrack test_racetrack \
	train_merge test_merge test_intersection \
	train_roundabout test_roundabout

help:
	@echo "🤖 KOMUTLAR:"
	@echo "  make setup           : venv oluşturur (varsa dokunmaz)."
	@echo "  make install         : requirements + dev tools kurar."
	@echo "  make format          : autoflake + black ile kodu düzeltir."
	@echo "  make lint            : flake8 ile PEP8/lint kontrolü."
	@echo "  make train_parking   : Parking eğitim."
	@echo "  make test_parking    : Parking test."
	@echo "  make train_racetrack : Racetrack eğitim."
	@echo "  make test_racetrack  : Racetrack test."
	@echo "  make train_merge     : Merge + Intersection eğitim (tek script)."
	@echo "  make test_merge      : Merge test."
	@echo "  make test_intersection : Intersection görselleştirme/test."
	@echo "  make clean           : venv ve cache temizler."

# ---------------------------------------------------------------------------
# Setup & Install
# ---------------------------------------------------------------------------
setup:
	@test -d $(VENV) || $(SYSTEM_PYTHON) -m venv $(VENV)
	@echo "Venv hazır. Aktif etmek için: source $(VENV)/bin/activate"

install: setup
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	# Dev tools (PEP8/format/lint)
	$(PIP) install black flake8 autoflake
	@echo "Kurulum tamam."

clean:
	rm -rf $(VENV)
	rm -rf .pytest_cache .coverage
	find . -type d -name "__pycache__" -print0 | xargs -0 rm -rf || true
	find . -type d -name "*.egg-info" -print0 | xargs -0 rm -rf || true
	@echo "Temizlik tamamlandı."

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
lint: setup
	@echo "🔍 Linting (flake8)..."
	$(FLAKE8) $(SRC) --count --statistics --max-line-length=88

format: setup
	@echo "✨ Autoflake + Black çalışıyor..."
	# Remove unused imports ONLY in our source files (safe list)
	$(AUTOFLAKE) --in-place --remove-all-unused-imports --remove-unused-variables $(SRC)
	# Format with Black (PEP8-ish style)
	$(BLACK) $(SRC)
	@echo "Format tamam."

# ---------------------------------------------------------------------------
# Parking
# ---------------------------------------------------------------------------
train_parking: setup
	@echo "Parking eğitimi başlıyor..."
	$(PY) parking_training.py

test_parking: setup
	@echo "Parking testi başlıyor..."
	$(PY) parking_test.py

# ---------------------------------------------------------------------------
# Racetrack
# ---------------------------------------------------------------------------
train_racetrack: setup
	@echo "Racetrack eğitimi başlıyor..."
	$(PY) train_racetrack.py

test_racetrack: setup
	@echo "Racetrack testi başlıyor..."
	$(PY) racetrack_test.py

# ---------------------------------------------------------------------------
# Merge & Intersection
# ---------------------------------------------------------------------------
train_merge: setup
	@echo "Merge + Intersection eğitimi başlıyor..."
	$(PY) merge+intersection_train.py

test_merge: setup
	@echo "Merge testi başlıyor..."
	$(PY) test_merge.py

test_intersection: setup
	@echo "Intersection test başlıyor..."
	$(PY) test_intersection.py

# ---------------------------------------------------------------------------
# Roundabout
# ---------------------------------------------------------------------------
train_roundabout: setup
	@echo "Roundabout eğitimi başlıyor..."
	$(PY) roundabout_training.py

test_roundabout: setup
	@echo "Roundabout test başlıyor..."
	$(PY) roundabout_test.py


# ---------------------------------------------------------------------------
# Highway v4 (Smart Aggressive)
# ---------------------------------------------------------------------------
train_highway: setup
	@echo "🚗 Highway v4 (Smart Aggressive) eğitimi başlıyor..."
	$(PY) highway_train.py

test_highway: setup
	@echo "🎬 Highway v4 testi başlıyor..."
	$(PY) highway_test.py