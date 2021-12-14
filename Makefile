NAME?=ocr-model

GPUS?=all
GPUS_OPTION=--gpus=$(GPUS)

CPUS?=none
ifeq ($(CPUS), none)
	CPUS_OPTION=
else
	CPUS_OPTION=--cpus=$(CPUS)
endif

.PHONY: all stop build run

all: stop build run

build:
	docker build \
	-t $(NAME) .

stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

run:
	docker run --rm -it \
		$(GPUS_OPTION) \
		$(CPUS_OPTION) \
		--net=host \
		--ipc=host \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		bash
