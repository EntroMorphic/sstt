CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma -march=native -Wall -Wextra
LDFLAGS = -lm

MNIST_URL   = https://ossci-datasets.s3.amazonaws.com/mnist
FASHION_URL = http://fashion-mnist.s3-website.eu-central-1.amazonaws.com
MNIST_FILES = train-images-idx3-ubyte train-labels-idx1-ubyte \
              t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte

# Core classifiers (best results, most useful starting points)
CORE = sstt_v2 sstt_bytecascade sstt_multidot sstt_diagnose

# All experiment binaries
ALL_EXPERIMENTS = $(CORE) \
	sstt_mvp sstt_geom sstt_fused_test \
	sstt_bytepacked sstt_pentary sstt_transitions \
	sstt_series sstt_eigenseries sstt_ensemble \
	sstt_hybrid sstt_softcascade sstt_perihelion \
	sstt_push sstt_tiled sstt_tpca

# Default: build the four most useful binaries
all: $(CORE)

# Build everything
experiments: $(ALL_EXPERIMENTS)

# ----------------------------------------------------------------
# Core classifiers
# ----------------------------------------------------------------
sstt_v2: sstt_v2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_bytecascade: sstt_bytecascade.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_multidot: sstt_multidot.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_diagnose: sstt_diagnose.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ----------------------------------------------------------------
# Exploration experiments
# ----------------------------------------------------------------
sstt_mvp: sstt_mvp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_geom: sstt_geom.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_bytepacked: sstt_bytepacked.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_pentary: sstt_pentary.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_transitions: sstt_transitions.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_series: sstt_series.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_eigenseries: sstt_eigenseries.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_ensemble: sstt_ensemble.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_hybrid: sstt_hybrid.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_softcascade: sstt_softcascade.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_perihelion: sstt_perihelion.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_push: sstt_push.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_tiled: sstt_tiled.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_tpca: sstt_tpca.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ----------------------------------------------------------------
# Fused kernel (C + optional ASM variant)
# ----------------------------------------------------------------
sstt_fused_test: sstt_fused_test.c sstt_fused_c.c sstt_fused.h
	$(CC) $(CFLAGS) -o $@ sstt_fused_test.c sstt_fused_c.c $(LDFLAGS)

# ASM variant (WIP — operand ordering bug under investigation)
sstt_fused_test_asm: sstt_fused_test.c sstt_fused.S sstt_fused.h
	$(CC) $(CFLAGS) -o $@ sstt_fused_test.c sstt_fused.S $(LDFLAGS)

# ----------------------------------------------------------------
# Data
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------
clean:
	rm -f $(ALL_EXPERIMENTS) sstt_fused_test sstt_fused_test_asm

cleanall: clean
	rm -rf data data-fashion

.PHONY: all experiments clean cleanall mnist fashion
