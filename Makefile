# Build and check the release on macOS. Nothing here loads or dispatches a GPU program.
#
#   make native-tools   the decoder wrapper and the native harnesses (clang from Xcode's command-line tools)
#   make examples       workflows 1-4: compile (CPU), simulate (CPU), and two receipt checks (evidence only)
#   make test           the retained tests (release/tests.txt in the research checkout chose them)
#   make check          all three
#   make platform       compare this machine with the measured one (it does not fail on a difference; it says what it bears on)
PYTHON ?= python3
CC = clang
CXX = clang++
NATIVE_CFLAGS ?= -O2
EXAMPLE_OUT ?= $(shell mktemp -d -t agxforge-example)/out

.PHONY: all native-tools examples test check platform
all: native-tools

native-tools: tools/agx3dis tools/agx3meta tools/libagx3dis.dylib tools/g17scanworker spike/accel/libaccel.dylib \
	tools/g17decodegen tools/g17specgen tools/g17twinrun

# Apple's G17 decoder, reached through GPUCompiler.framework at run time (the compiler's release checks use it)
tools/agx3dis: tools/agx3dis.c tools/agx3remap.h tools/agx3renumber.h
	$(CC) $(NATIVE_CFLAGS) -o $@ $<

tools/libagx3dis.dylib: tools/agx3dislib.c tools/agx3remap.h tools/agx3renumber.h
	$(CC) $(NATIVE_CFLAGS) -dynamiclib -o $@ $<

tools/agx3meta: tools/agx3meta.c tools/agx3renumber.h
	$(CC) $(NATIVE_CFLAGS) -o $@ $<

tools/g17scanworker: tools/g17scanworker.m tools/g17scanstorage.h
	$(CC) -fobjc-arc $(NATIVE_CFLAGS) -Wall -Wextra -Werror -framework Foundation -framework Metal -o $@ $<

# the chained-token executor through Metal (one command buffer a token; MM 25.138)
tools/g17decodegen: tools/g17decodegen.m tools/g17gpulock.h
	$(CC) -fobjc-arc $(NATIVE_CFLAGS) -I tools -framework Foundation -framework Metal -o $@ $<

tools/g17specgen: tools/g17specgen.m tools/g17gpulock.h
	$(CC) -fobjc-arc $(NATIVE_CFLAGS) -Wall -Wextra -Werror -I tools -framework Foundation -framework Metal -o $@ $<

# the matched study's kernel harness: a bundle and its Apple-compiled twin on the same buffers (MM 25.211)
tools/g17twinrun: tools/g17twinrun.m tools/g17gpulock.h
	$(CC) -fobjc-arc $(NATIVE_CFLAGS) -Wall -Wextra -Werror -I tools -framework Foundation -framework Metal -o $@ $<

spike/accel/libaccel.dylib: spike/accel/accel.mm tools/g17gpulock.h
	$(CXX) $(NATIVE_CFLAGS) -dynamiclib -fobjc-arc -framework Foundation -framework Metal $< -o $@

examples: native-tools
	$(PYTHON) examples/tensor_17x19x16.py $(EXAMPLE_OUT)
	$(PYTHON) examples/tensor_feed_rule.py
	$(PYTHON) examples/decode_study.py
	$(PYTHON) examples/native_inference.py

test: native-tools
	$(PYTHON) tools/release_tests.py

check: examples test

platform:
	$(PYTHON) tools/g17platform.py --check
