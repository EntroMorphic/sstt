CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma -march=native -Wall -Wextra
LDFLAGS = -lm
BUILD   = build

MNIST_URL   = https://ossci-datasets.s3.amazonaws.com/mnist
FASHION_URL = https://fashion-mnist.s3-website.eu-central-1.amazonaws.com
KMNIST_URL  = http://codh.rois.ac.jp/kmnist/dataset/kmnist
MNIST_FILES = train-images-idx3-ubyte train-labels-idx1-ubyte \
              t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte

# ================================================================
# Grouped targets
# ================================================================

# Core: publication-ready classifiers (default build)
CORE = $(BUILD)/sstt_topo9_val \
       $(BUILD)/sstt_bytecascade \
       $(BUILD)/sstt_router_v1 \
       $(BUILD)/sstt_v2 \
       $(BUILD)/sstt_kinvariance \
       $(BUILD)/sstt_ann_baseline \
       $(BUILD)/sstt_hybrid_retrieval \
       $(BUILD)/sstt_mtfp

# Analysis: diagnostic and validation tools
ANALYSIS = $(BUILD)/sstt_diagnose \
           $(BUILD)/sstt_error_profile \
           $(BUILD)/sstt_confidence_map \
           $(BUILD)/sstt_validate \
           $(BUILD)/sstt_kdilute \
           $(BUILD)/sstt_gauss_map \
           $(BUILD)/sstt_gauss_delta

# Ablation: topo1-topo9 series + variants
ABLATION = $(BUILD)/sstt_topo \
           $(BUILD)/sstt_topo2 $(BUILD)/sstt_topo3 $(BUILD)/sstt_topo4 \
           $(BUILD)/sstt_topo5 $(BUILD)/sstt_topo6 $(BUILD)/sstt_topo7 \
           $(BUILD)/sstt_topo8 $(BUILD)/sstt_topo9 \
           $(BUILD)/sstt_topo9_val_5trit \
           $(BUILD)/sstt_topo9_val_5trit_grid

# CIFAR-10: boundary experiments
CIFAR10 = $(BUILD)/sstt_cifar10 \
          $(BUILD)/sstt_cifar10_5trit_quick \
          $(BUILD)/sstt_cifar10_adaptive \
          $(BUILD)/sstt_cifar10_binary $(BUILD)/sstt_cifar10_binary_dig \
          $(BUILD)/sstt_cifar10_cascade_gauss \
          $(BUILD)/sstt_cifar10_correlate $(BUILD)/sstt_cifar10_curvature \
          $(BUILD)/sstt_cifar10_dual $(BUILD)/sstt_cifar10_edgemask \
          $(BUILD)/sstt_cifar10_flat $(BUILD)/sstt_cifar10_full \
          $(BUILD)/sstt_cifar10_full_lagrangian $(BUILD)/sstt_cifar10_fused \
          $(BUILD)/sstt_cifar10_gauss $(BUILD)/sstt_cifar10_grad \
          $(BUILD)/sstt_cifar10_grid_tracer $(BUILD)/sstt_cifar10_lagrangian \
          $(BUILD)/sstt_cifar10_modec $(BUILD)/sstt_cifar10_moe \
          $(BUILD)/sstt_cifar10_mt4 $(BUILD)/sstt_cifar10_mt4vote \
          $(BUILD)/sstt_cifar10_mt7vote $(BUILD)/sstt_cifar10_multi_eye \
          $(BUILD)/sstt_cifar10_multiscale $(BUILD)/sstt_cifar10_propagate \
          $(BUILD)/sstt_cifar10_qmad $(BUILD)/sstt_cifar10_rawdot \
          $(BUILD)/sstt_cifar10_router $(BUILD)/sstt_cifar10_stack \
          $(BUILD)/sstt_cifar10_stereo $(BUILD)/sstt_cifar10_stereo_stack \
          $(BUILD)/sstt_cifar10_ternvbin $(BUILD)/sstt_cifar10_tracer \
          $(BUILD)/sstt_cifar10_unified_geom $(BUILD)/sstt_cifar10_why \
          $(BUILD)/sstt_benchmark_cifar10

# Archive: exploratory and superseded experiments
ARCHIVE = $(BUILD)/sstt_mvp $(BUILD)/sstt_geom \
          $(BUILD)/sstt_bytepacked $(BUILD)/sstt_pentary \
          $(BUILD)/sstt_transitions $(BUILD)/sstt_series \
          $(BUILD)/sstt_eigenseries $(BUILD)/sstt_ensemble \
          $(BUILD)/sstt_hybrid $(BUILD)/sstt_softcascade \
          $(BUILD)/sstt_perihelion $(BUILD)/sstt_push \
          $(BUILD)/sstt_tiled $(BUILD)/sstt_tpca \
          $(BUILD)/sstt_oracle $(BUILD)/sstt_oracle_v2 $(BUILD)/sstt_oracle_v3 \
          $(BUILD)/sstt_parallel $(BUILD)/sstt_navier_stokes \
          $(BUILD)/sstt_taylor_jet $(BUILD)/sstt_taylor_specialist \
          $(BUILD)/sstt_scale_pro $(BUILD)/sstt_scale_hierarchical \
          $(BUILD)/sstt_scale_224 $(BUILD)/sstt_scale_brute \
          $(BUILD)/sstt_vote_route $(BUILD)/sstt_delta_map \
          $(BUILD)/sstt_dual_hot_map $(BUILD)/sstt_fashion_stereo \
          $(BUILD)/sstt_multidot \
          $(BUILD)/sstt_bag_positions $(BUILD)/sstt_bag_topo9 \
          $(BUILD)/sstt_multi_threshold_topo9 \
          $(BUILD)/sstt_deep_ternary_fashion

ALL_EXPERIMENTS = $(CORE) $(ANALYSIS) $(ABLATION) $(CIFAR10) $(ARCHIVE) \
                  $(BUILD)/sstt_fused_test $(BUILD)/sstt_fused_test_asm \
                  $(BUILD)/sstt_router_hardened_test

# ================================================================
# Top-level targets
# ================================================================

# Default: build the 4 core classifiers
all: $(BUILD) $(CORE)

# Named group targets
core: $(BUILD) $(CORE)
analysis: $(BUILD) $(ANALYSIS)
ablation: $(BUILD) $(ABLATION)
cifar10-experiments: $(BUILD) $(CIFAR10)
archive: $(BUILD) $(ARCHIVE)

# Build everything
experiments: $(BUILD) $(ALL_EXPERIMENTS)

# Reproduce headline results (build + download + run)
reproduce: $(BUILD) $(CORE) mnist
	@echo "=== Reproducing SSTT headline results ==="
	@echo ""
	@echo "--- Bytepacked Cascade (expect ~96.28%) ---"
	./$(BUILD)/sstt_bytecascade
	@echo ""
	@echo "--- Topo9 Val/Holdout (expect ~97.27%) ---"
	./$(BUILD)/sstt_topo9_val

$(BUILD):
	mkdir -p $(BUILD)

# Order-only prerequisite: all targets need build/ to exist
$(ALL_EXPERIMENTS): | $(BUILD)

# ================================================================
# Core classifiers (src/core/)
# ================================================================

$(BUILD)/sstt_topo9_val: src/core/sstt_topo9_val.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_bytecascade: src/core/sstt_bytecascade.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_router_v1: src/core/sstt_router_v1.c src/core/sstt_fused.S src/core/sstt_fused.h
	$(CC) $(CFLAGS) -Isrc/core -o $@ src/core/sstt_router_v1.c src/core/sstt_fused.S $(LDFLAGS)

$(BUILD)/sstt_v2: src/core/sstt_v2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_fused_test: src/core/sstt_fused_test.c src/core/sstt_fused_c.c src/core/sstt_fused.h
	$(CC) $(CFLAGS) -o $@ src/core/sstt_fused_test.c src/core/sstt_fused_c.c $(LDFLAGS)

$(BUILD)/sstt_fused_test_asm: src/core/sstt_fused_test.c src/core/sstt_fused.S src/core/sstt_fused.h
	$(CC) $(CFLAGS) -o $@ src/core/sstt_fused_test.c src/core/sstt_fused.S $(LDFLAGS)

$(BUILD)/sstt_router_hardened_test: src/core/sstt_router_hardened_test.c src/core/sstt_fused.S src/core/sstt_fused.h
	$(CC) $(CFLAGS) -Isrc/core -o $@ src/core/sstt_router_hardened_test.c src/core/sstt_fused.S $(LDFLAGS)

$(BUILD)/sstt_kinvariance: src/core/sstt_kinvariance.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_ann_baseline: src/core/sstt_ann_baseline.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_hybrid_retrieval: src/core/sstt_hybrid_retrieval.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_mtfp: src/core/sstt_mtfp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ================================================================
# Analysis tools (src/analysis/)
# ================================================================

$(BUILD)/sstt_diagnose: src/analysis/sstt_diagnose.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_error_profile: src/analysis/sstt_error_profile.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_confidence_map: src/analysis/sstt_confidence_map.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_validate: src/analysis/sstt_validate.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_kdilute: src/analysis/sstt_kdilute.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_gauss_map: src/analysis/sstt_gauss_map.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_gauss_delta: src/analysis/sstt_gauss_delta.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ================================================================
# Ablation series (src/ablation/)
# ================================================================

$(BUILD)/sstt_topo: src/ablation/sstt_topo.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo2: src/ablation/sstt_topo2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo3: src/ablation/sstt_topo3.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo4: src/ablation/sstt_topo4.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo5: src/ablation/sstt_topo5.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo6: src/ablation/sstt_topo6.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo7: src/ablation/sstt_topo7.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo8: src/ablation/sstt_topo8.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo9: src/ablation/sstt_topo9.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo9_val_5trit: src/ablation/sstt_topo9_val_5trit.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo9_val_5trit_grid: src/ablation/sstt_topo9_val_5trit_grid.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ================================================================
# CIFAR-10 experiments (src/cifar10/)
# ================================================================

$(BUILD)/sstt_cifar10: src/cifar10/sstt_cifar10.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_benchmark_cifar10: src/cifar10/sstt_benchmark_cifar10.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Pattern rule for all sstt_cifar10_* variants
$(BUILD)/sstt_cifar10_%: src/cifar10/sstt_cifar10_%.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ================================================================
# Archive experiments (src/archive/)
# ================================================================

# Pattern rule: single-file archive experiments
$(BUILD)/sstt_%: src/archive/sstt_%.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ================================================================
# Data download
# ================================================================

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

kmnist: $(addprefix data-kmnist/, $(MNIST_FILES))

data-kmnist/%: data-kmnist/%.gz
	gunzip -k $<

data-kmnist/%.gz:
	mkdir -p data-kmnist
	curl -sS -o $@ $(KMNIST_URL)/$*.gz

# ================================================================
# Cleanup
# ================================================================

clean:
	rm -rf $(BUILD)

cleanall: clean
	rm -rf data data-fashion data-cifar10 data-kmnist

.PHONY: all core analysis ablation cifar10-experiments archive \
        experiments reproduce clean cleanall mnist fashion kmnist
