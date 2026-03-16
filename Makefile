CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma -march=native -Wall -Wextra
LDFLAGS = -lm
SRC     = src

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
sstt_v2: $(SRC)/sstt_v2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_bytecascade: $(SRC)/sstt_bytecascade.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_multidot: $(SRC)/sstt_multidot.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_diagnose: $(SRC)/sstt_diagnose.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ----------------------------------------------------------------
# Exploration experiments
# ----------------------------------------------------------------
sstt_mvp: $(SRC)/sstt_mvp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_geom: $(SRC)/sstt_geom.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_bytepacked: $(SRC)/sstt_bytepacked.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_pentary: $(SRC)/sstt_pentary.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_transitions: $(SRC)/sstt_transitions.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_series: $(SRC)/sstt_series.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_eigenseries: $(SRC)/sstt_eigenseries.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_ensemble: $(SRC)/sstt_ensemble.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_hybrid: $(SRC)/sstt_hybrid.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_softcascade: $(SRC)/sstt_softcascade.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_perihelion: $(SRC)/sstt_perihelion.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_push: $(SRC)/sstt_push.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_tiled: $(SRC)/sstt_tiled.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_tpca: $(SRC)/sstt_tpca.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_oracle: $(SRC)/sstt_oracle.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_oracle_v2: $(SRC)/sstt_oracle_v2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_parallel: $(SRC)/sstt_parallel.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_oracle_v3: $(SRC)/sstt_oracle_v3.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ----------------------------------------------------------------
# Fused kernel (C + optional ASM variant)
# ----------------------------------------------------------------
sstt_fused_test: $(SRC)/sstt_fused_test.c $(SRC)/sstt_fused_c.c $(SRC)/sstt_fused.h
	$(CC) $(CFLAGS) -o $@ $(SRC)/sstt_fused_test.c $(SRC)/sstt_fused_c.c $(LDFLAGS)

# ASM variant (WIP — operand ordering bug under investigation)
sstt_fused_test_asm: $(SRC)/sstt_fused_test.c $(SRC)/sstt_fused.S $(SRC)/sstt_fused.h
	$(CC) $(CFLAGS) -o $@ $(SRC)/sstt_fused_test.c $(SRC)/sstt_fused.S $(LDFLAGS)

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
