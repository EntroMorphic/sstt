CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma -march=native -Wall -Wextra
LDFLAGS = -lm
SRC     = src
BUILD   = build

MNIST_URL   = https://ossci-datasets.s3.amazonaws.com/mnist
FASHION_URL = https://fashion-mnist.s3-website.eu-central-1.amazonaws.com
MNIST_FILES = train-images-idx3-ubyte train-labels-idx1-ubyte \
              t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte

# Core classifiers (best results, most useful starting points)
CORE = $(BUILD)/sstt_v2 $(BUILD)/sstt_bytecascade $(BUILD)/sstt_multidot $(BUILD)/sstt_diagnose

# All experiment binaries
ALL_EXPERIMENTS = $(CORE) \
	$(BUILD)/sstt_mvp $(BUILD)/sstt_geom $(BUILD)/sstt_fused_test \
	$(BUILD)/sstt_bytepacked $(BUILD)/sstt_pentary $(BUILD)/sstt_transitions \
	$(BUILD)/sstt_series $(BUILD)/sstt_eigenseries $(BUILD)/sstt_ensemble \
	$(BUILD)/sstt_hybrid $(BUILD)/sstt_softcascade $(BUILD)/sstt_perihelion \
	$(BUILD)/sstt_push $(BUILD)/sstt_tiled $(BUILD)/sstt_tpca \
	$(BUILD)/sstt_oracle $(BUILD)/sstt_oracle_v2 $(BUILD)/sstt_oracle_v3 $(BUILD)/sstt_parallel \
	$(BUILD)/sstt_topo $(BUILD)/sstt_topo2 $(BUILD)/sstt_topo3 $(BUILD)/sstt_topo4 \
	$(BUILD)/sstt_topo5 $(BUILD)/sstt_topo6 $(BUILD)/sstt_topo7 $(BUILD)/sstt_topo8 $(BUILD)/sstt_topo9 \
	$(BUILD)/sstt_topo9_val $(BUILD)/sstt_topo9_val_5trit $(BUILD)/sstt_gauss_delta $(BUILD)/sstt_kdilute \
	$(BUILD)/sstt_navier_stokes $(BUILD)/sstt_taylor_jet $(BUILD)/sstt_taylor_specialist \
	$(BUILD)/sstt_scale_pro $(BUILD)/sstt_scale_hierarchical $(BUILD)/sstt_scale_224 \
	$(BUILD)/sstt_scale_brute \
	$(BUILD)/sstt_error_profile $(BUILD)/sstt_gauss_map $(BUILD)/sstt_confidence_map \
	$(BUILD)/sstt_vote_route $(BUILD)/sstt_validate $(BUILD)/sstt_delta_map \
	$(BUILD)/sstt_dual_hot_map $(BUILD)/sstt_router_v1 $(BUILD)/sstt_fashion_stereo \
	$(BUILD)/sstt_cifar10 $(BUILD)/sstt_cifar10_flat $(BUILD)/sstt_cifar10_grad $(BUILD)/sstt_cifar10_mt4 \
	$(BUILD)/sstt_cifar10_full $(BUILD)/sstt_cifar10_stack $(BUILD)/sstt_cifar10_mt4vote \
	$(BUILD)/sstt_cifar10_modec $(BUILD)/sstt_cifar10_why $(BUILD)/sstt_cifar10_rawdot \
	$(BUILD)/sstt_cifar10_moe $(BUILD)/sstt_cifar10_qmad $(BUILD)/sstt_cifar10_propagate \
	$(BUILD)/sstt_cifar10_mt7vote $(BUILD)/sstt_cifar10_unified_geom \
	$(BUILD)/sstt_cifar10_cascade_gauss $(BUILD)/sstt_cifar10_gauss \
	$(BUILD)/sstt_cifar10_stereo $(BUILD)/sstt_cifar10_stereo_stack \
	$(BUILD)/sstt_cifar10_adaptive $(BUILD)/sstt_cifar10_binary $(BUILD)/sstt_cifar10_binary_dig \
	$(BUILD)/sstt_cifar10_correlate $(BUILD)/sstt_cifar10_curvature \
	$(BUILD)/sstt_cifar10_dual $(BUILD)/sstt_cifar10_edgemask \
	$(BUILD)/sstt_cifar10_full_lagrangian $(BUILD)/sstt_cifar10_fused \
	$(BUILD)/sstt_cifar10_grid_tracer $(BUILD)/sstt_cifar10_lagrangian \
	$(BUILD)/sstt_cifar10_multiscale $(BUILD)/sstt_cifar10_router \
	$(BUILD)/sstt_cifar10_ternvbin $(BUILD)/sstt_cifar10_tracer \
	$(BUILD)/sstt_benchmark_cifar10 \
	$(BUILD)/sstt_bag_positions

# Default: build the four most useful binaries
all: $(BUILD) $(CORE)

# Build everything
experiments: $(BUILD) $(ALL_EXPERIMENTS)

$(BUILD):
	mkdir -p $(BUILD)

# All targets depend on the build directory existing (order-only: no rebuild on dir timestamp)
$(ALL_EXPERIMENTS) $(BUILD)/sstt_fused_test $(BUILD)/sstt_fused_test_asm: | $(BUILD)

# ----------------------------------------------------------------
# Core classifiers
# ----------------------------------------------------------------
$(BUILD)/sstt_v2: $(SRC)/sstt_v2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_bytecascade: $(SRC)/sstt_bytecascade.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_multidot: $(SRC)/sstt_multidot.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_diagnose: $(SRC)/sstt_diagnose.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ----------------------------------------------------------------
# Exploration experiments
# ----------------------------------------------------------------
$(BUILD)/sstt_mvp: $(SRC)/sstt_mvp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_geom: $(SRC)/sstt_geom.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_bytepacked: $(SRC)/sstt_bytepacked.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_pentary: $(SRC)/sstt_pentary.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_transitions: $(SRC)/sstt_transitions.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_series: $(SRC)/sstt_series.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_eigenseries: $(SRC)/sstt_eigenseries.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_ensemble: $(SRC)/sstt_ensemble.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_hybrid: $(SRC)/sstt_hybrid.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_softcascade: $(SRC)/sstt_softcascade.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_perihelion: $(SRC)/sstt_perihelion.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_push: $(SRC)/sstt_push.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_tiled: $(SRC)/sstt_tiled.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_tpca: $(SRC)/sstt_tpca.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_oracle: $(SRC)/sstt_oracle.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_oracle_v2: $(SRC)/sstt_oracle_v2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_parallel: $(SRC)/sstt_parallel.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_oracle_v3: $(SRC)/sstt_oracle_v3.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo: $(SRC)/sstt_topo.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo2: $(SRC)/sstt_topo2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo3: $(SRC)/sstt_topo3.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo4: $(SRC)/sstt_topo4.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo5: $(SRC)/sstt_topo5.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo6: $(SRC)/sstt_topo6.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo7: $(SRC)/sstt_topo7.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo8: $(SRC)/sstt_topo8.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo9: $(SRC)/sstt_topo9.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo9_val: $(SRC)/sstt_topo9_val.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_topo9_val_5trit: $(SRC)/sstt_topo9_val_5trit.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_gauss_delta: $(SRC)/sstt_gauss_delta.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_kdilute: $(SRC)/sstt_kdilute.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ----------------------------------------------------------------
# Topology / analysis experiments (previously missing targets)
# ----------------------------------------------------------------
$(BUILD)/sstt_error_profile: $(SRC)/sstt_error_profile.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_gauss_map: $(SRC)/sstt_gauss_map.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_confidence_map: $(SRC)/sstt_confidence_map.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_vote_route: $(SRC)/sstt_vote_route.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_validate: $(SRC)/sstt_validate.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_delta_map: $(SRC)/sstt_delta_map.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_dual_hot_map: $(SRC)/sstt_dual_hot_map.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_router_v1: $(SRC)/sstt_router_v1.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_fashion_stereo: $(SRC)/sstt_fashion_stereo.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_scale_224: $(SRC)/sstt_scale_224.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_scale_brute: $(SRC)/sstt_scale_brute.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_benchmark_cifar10: $(SRC)/sstt_benchmark_cifar10.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ----------------------------------------------------------------
# Field-theoretic and scaling experiments (contributions 57-59)
# ----------------------------------------------------------------
$(BUILD)/sstt_navier_stokes: $(SRC)/sstt_navier_stokes.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_taylor_jet: $(SRC)/sstt_taylor_jet.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_taylor_specialist: $(SRC)/sstt_taylor_specialist.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_scale_pro: $(SRC)/sstt_scale_pro.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_scale_hierarchical: $(SRC)/sstt_scale_hierarchical.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ----------------------------------------------------------------
# CIFAR-10 experiments
# ----------------------------------------------------------------
$(BUILD)/sstt_cifar10: $(SRC)/sstt_cifar10.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/sstt_cifar10_flat $(BUILD)/sstt_cifar10_grad \
$(BUILD)/sstt_cifar10_mt4 $(BUILD)/sstt_cifar10_full $(BUILD)/sstt_cifar10_stack \
$(BUILD)/sstt_cifar10_mt4vote $(BUILD)/sstt_cifar10_modec $(BUILD)/sstt_cifar10_why \
$(BUILD)/sstt_cifar10_rawdot $(BUILD)/sstt_cifar10_moe $(BUILD)/sstt_cifar10_qmad \
$(BUILD)/sstt_cifar10_propagate $(BUILD)/sstt_cifar10_mt7vote \
$(BUILD)/sstt_cifar10_unified_geom $(BUILD)/sstt_cifar10_cascade_gauss \
$(BUILD)/sstt_cifar10_gauss $(BUILD)/sstt_cifar10_stereo $(BUILD)/sstt_cifar10_stereo_stack \
$(BUILD)/sstt_cifar10_adaptive $(BUILD)/sstt_cifar10_binary $(BUILD)/sstt_cifar10_binary_dig \
$(BUILD)/sstt_cifar10_correlate $(BUILD)/sstt_cifar10_curvature \
$(BUILD)/sstt_cifar10_dual $(BUILD)/sstt_cifar10_edgemask \
$(BUILD)/sstt_cifar10_full_lagrangian $(BUILD)/sstt_cifar10_fused \
$(BUILD)/sstt_cifar10_grid_tracer $(BUILD)/sstt_cifar10_lagrangian \
$(BUILD)/sstt_cifar10_multiscale $(BUILD)/sstt_cifar10_router \
$(BUILD)/sstt_cifar10_ternvbin $(BUILD)/sstt_cifar10_tracer: $(BUILD)/sstt_cifar10_%: $(SRC)/sstt_cifar10_%.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ----------------------------------------------------------------
# Fused kernel (C + optional ASM variant)
# ----------------------------------------------------------------
$(BUILD)/sstt_fused_test: $(SRC)/sstt_fused_test.c $(SRC)/sstt_fused_c.c $(SRC)/sstt_fused.h
	$(CC) $(CFLAGS) -o $@ $(SRC)/sstt_fused_test.c $(SRC)/sstt_fused_c.c $(LDFLAGS)

# ASM variant (WIP — operand ordering bug under investigation)
$(BUILD)/sstt_fused_test_asm: $(SRC)/sstt_fused_test.c $(SRC)/sstt_fused.S $(SRC)/sstt_fused.h
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
	rm -rf $(BUILD)

cleanall: clean
	rm -rf data data-fashion data-cifar10

.PHONY: all experiments clean cleanall mnist fashion
