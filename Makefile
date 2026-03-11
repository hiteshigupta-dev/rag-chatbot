# RAG-based Tonton FAQ Chatbot — Makefile
# Run from project root (directory containing app.py and requirements.txt)

.PHONY: install build-index run clean format help

# Default target
help:
	@echo "RAG Tonton FAQ Chatbot — available targets:"
	@echo "  make install     — install Python dependencies"
	@echo "  make build-index — build FAISS index from data/FAQ.pdf"
	@echo "  make run         — start Streamlit app"
	@echo "  make clean       — remove cache and temporary files"
	@echo "  make format      — run Black formatter (optional)"

# Install dependencies from requirements.txt
install:
	pip install -r requirements.txt

# Build FAISS index (run after adding data/faq.pdf or data/FAQ.pdf)
build-index:
	python build_index.py

# Run the Streamlit chat application
run:
	streamlit run app.py

# Remove cache and temporary files (Python for Windows/Unix portability)
clean:
	python -c "import pathlib, shutil; cache=pathlib.Path('cache'); [f.unlink() for f in cache.glob('*.json') if f.is_file()]; [shutil.rmtree(d) for d in pathlib.Path('.').rglob('__pycache__') if d.is_dir()]"
	@echo "Cache and temporary files cleaned."

# Optional: format code with Black (install with: pip install black)
format:
	python -m black app.py build_index.py src/
	@echo "Formatting complete."
