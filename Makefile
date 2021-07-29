dist:
	python setup.py build sdist bdist_wheel

build:
	python setup.py build_ext --inplace

clean:
	find . -name "dist" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "builds" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*.so" | xargs rm -rf
	find . -name "*.egg-info" | xargs rm -rf
	find . -name ".ipynb_checkpoints" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf

