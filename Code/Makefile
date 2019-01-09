all:
	python setup_cython.py build_ext --inplace

clean:
	rm -v lib_*.so lib_*.c lib_*.html lib_*.pyc

tests:
	nosetests -w Tests -v

html:
	for x in lib*pyx; do echo $$x; cython -a $$x; done
