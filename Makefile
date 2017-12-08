all:
	cd bin && find . -type f ! -name '*.ocl' ! -name 'Makefile' -delete
	cd build && rm -rf *
	cd build && cmake ../src
	cd build && make

cleanup:
	cd bin && find . -type f ! -name '*.ocl' ! -name 'Makefile' -delete
	cd build && rm -rf *