export COLOREDLOGS_LOG_FORMAT=%(asctime)s - %(message)s
CONFIG=size.ini

all:
	@echo "dev/clean/collect"

dev:
	python nnf.py configs/$(CONFIG)

clean:
	rm -rf programs/
