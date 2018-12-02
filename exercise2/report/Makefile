# Zu verwendende Shell umstellen, damit Brace Extensions funktionieren.
SHELL=/bin/bash

TEX_FILES=$(wildcard *.tex)
TARGET=Exercise_01

$(TARGET).pdf: $(TEX_FILES)
	xelatex $(TARGET)
	xelatex $(TARGET)

clean:
	$(RM) *.{aux,log,out}
.PHONY: clean

distclean:
	$(RM) $(TARGET).pdf *.{aux,log,out}
.PHONY: distclean
