# Convenience targets. Everything also works as `python -m longeval_starter ...`

PY ?= python

SNAPSHOTS := snapshot-1 snapshot-2 snapshot-3

.PHONY: help index retrieve evaluate all clean

help:
	@echo "make index       - build an index for every snapshot"
	@echo "make retrieve    - run the pipeline and write TREC runs for every snapshot"
	@echo "make evaluate    - evaluate on the training qrels of snapshot-1"
	@echo "make all         - index + retrieve + evaluate"
	@echo "make clean       - remove indexes/ and runs/"

index:
	@for s in $(SNAPSHOTS); do \
		$(PY) -m longeval_starter index --snapshot $$s ; \
	done

retrieve:
	@for s in $(SNAPSHOTS); do \
		$(PY) -m longeval_starter retrieve --snapshot $$s ; \
	done

evaluate:
	$(PY) -m longeval_starter evaluate --snapshot snapshot-1

all: index evaluate retrieve

clean:
	rm -rf indexes/ runs/
