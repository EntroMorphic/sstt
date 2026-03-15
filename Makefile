CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma -march=native -Wall -Wextra
LDFLAGS = -lm

MNIST_URL   = https://ossci-datasets.s3.amazonaws.com/mnist
FASHION_URL = http://fashion-mnist.s3-website.eu-central-1.amazonaws.com
MNIST_FILES = train-images-idx3-ubyte train-labels-idx1-ubyte \
              t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte

all: sstt_mvp sstt_geom sstt_v2 sstt_fused_test

sstt_mvp: sstt_mvp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_geom: sstt_geom.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_v2: sstt_v2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_fused_test: sstt_fused_test.c sstt_fused_c.c sstt_fused.h
	$(CC) $(CFLAGS) -o $@ sstt_fused_test.c sstt_fused_c.c $(LDFLAGS)

# ASM variant (WIP — operand ordering bug under investigation)
sstt_fused_test_asm: sstt_fused_test.c sstt_fused.S sstt_fused.h
	$(CC) $(CFLAGS) -o $@ sstt_fused_test.c sstt_fused.S $(LDFLAGS)

mnist: $(addprefix data/, $(MNIST_FILES))

data/%: data/%.gz
	gunzip -k $<

data/%.gz:
	mkdir -p data
	curl -sS -o $@ $(MNIST_URL)/$*.gz

fashion: $(addprefix data-fashion/, $(MNIST_FILES))

data-fashion/%: data-fashion/%.gz
	gunzip -k $<

data-fashion/%.gz:
	mkdir -p data-fashion
	curl -sS -o $@ $(FASHION_URL)/$*.gz

clean:
	rm -f sstt_mvp sstt_geom sstt_v2 sstt_fused_test

cleanall: clean
	rm -rf data data-fashion

.PHONY: all clean cleanall mnist fashion
