CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma -march=native -Wall -Wextra
LDFLAGS = -lm
SRC     = src

MNIST_URL   = https://ossci-datasets.s3.amazonaws.com/mnist
FASHION_URL = https://fashion-mnist.s3-website.eu-central-1.amazonaws.com
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
	sstt_push sstt_tiled sstt_tpca \
	sstt_oracle sstt_oracle_v2 sstt_oracle_v3 sstt_parallel \
	sstt_topo sstt_topo2 sstt_topo3 sstt_topo4 sstt_topo5 sstt_topo6 sstt_topo7 sstt_topo8 sstt_topo9 \
	sstt_topo9_val sstt_gauss_delta sstt_cifar10 sstt_kdilute

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

sstt_topo: $(SRC)/sstt_topo.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_topo2: $(SRC)/sstt_topo2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_topo3: $(SRC)/sstt_topo3.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_topo4: $(SRC)/sstt_topo4.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_topo5: $(SRC)/sstt_topo5.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_topo6: $(SRC)/sstt_topo6.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_topo7: $(SRC)/sstt_topo7.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_topo8: $(SRC)/sstt_topo8.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_topo9: $(SRC)/sstt_topo9.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_topo9_val: $(SRC)/sstt_topo9_val.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_gauss_delta: $(SRC)/sstt_gauss_delta.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_cifar10: $(SRC)/sstt_cifar10.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sstt_kdilute: $(SRC)/sstt_kdilute.c
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

# SHA-256 checksums for uncompressed IDX files
SHA256_train-images-idx3-ubyte = ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db
SHA256_train-labels-idx1-ubyte = 65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5
SHA256_t10k-images-idx3-ubyte  = 0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7
SHA256_t10k-labels-idx1-ubyte  = ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2

SHA256F_train-images-idx3-ubyte = c59f468a2f672dc815687fe0f83887768d799fd8a3f3276145d20f83aa44d888
SHA256F_train-labels-idx1-ubyte = bad3541b69d912435c50bb6ba87bec294ff4f6a2e1246121d8633921760443d9
SHA256F_t10k-images-idx3-ubyte  = 5b4141f0afbad91edebe8549f8fcffe087ea10ca49f1dbef5c9a5cd8815ce37b
SHA256F_t10k-labels-idx1-ubyte  = 0402a96d92fd2663957122ceb108a494c5af83dab82d92729df917d7dec38c34

mnist: $(addprefix data/, $(MNIST_FILES))

data/%: data/%.gz
	gunzip -k $<
	@echo "$(SHA256_$*)  $@" | sha256sum -c - || (rm -f $@ && exit 1)

data/%.gz:
	mkdir -p data
	curl -sS -o $@ $(MNIST_URL)/$*.gz

fashion: $(addprefix data-fashion/, $(MNIST_FILES))

data-fashion/%: data-fashion/%.gz
	gunzip -k $<
	@echo "$(SHA256F_$*)  $@" | sha256sum -c - || (rm -f $@ && exit 1)

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
