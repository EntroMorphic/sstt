CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma -march=native -Wall -Wextra
LDFLAGS = -lm

MNIST_URL   = https://ossci-datasets.s3.amazonaws.com/mnist
MNIST_FILES = train-images-idx3-ubyte train-labels-idx1-ubyte \
              t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte

all: sstt_mvp sstt_geom

sstt_mvp: sstt_mvp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_geom: sstt_geom.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

mnist: $(addprefix data/, $(MNIST_FILES))

data/%: data/%.gz
	gunzip -k $<

data/%.gz:
	mkdir -p data
	curl -sS -o $@ $(MNIST_URL)/$*.gz

clean:
	rm -f sstt_mvp sstt_geom

cleanall: clean
	rm -rf data

.PHONY: all clean cleanall mnist
