CC = cc
CFLAGS = -O3 -Wall -Wextra -std=c11 -I.

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK
    LDFLAGS = -framework Accelerate -lm
else
    CFLAGS += -DUSE_BLAS
    LDFLAGS = -lopenblas -lm
endif

.PHONY: all clean

all: train_janus_15m infer_janus15m

train_janus_15m: train_janus_15m.c notorch.c notorch.h
	$(CC) $(CFLAGS) -o $@ train_janus_15m.c notorch.c $(LDFLAGS)
	@echo "built: $@"

infer_janus15m: infer_janus15m.c notorch.c notorch.h
	$(CC) $(CFLAGS) -o $@ infer_janus15m.c notorch.c $(LDFLAGS)
	@echo "built: $@"

clean:
	rm -f train_janus_15m infer_janus15m *.o *.bin.bak
